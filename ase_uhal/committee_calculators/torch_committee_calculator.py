from .base_committee_calculator import BaseCommitteeCalculator
from abc import ABCMeta, abstractmethod
import numpy as np

class TorchCommitteeCalculator(BaseCommitteeCalculator, metaclass=ABCMeta):
    # Can be the result of tensor.get_device() for GPU-based operation, otherwise should be set to "cpu"
    # e.g. for MACE models:
    # self.torch_device = self.model.atomic_numbers.get_device()
    # if self.torch_device < 0:
    #     self.torch_device = "cpu"
    torch_device = None

    def __init__(self, *args, **kwargs):
        assert self.torch_device is not None, "self.torch_device should be set by subclasses before calling TorchCommitteeCalculator.__init__()"

        import torch
        self.torch = torch

        desc_len = self._get_descriptor_length()

        super().__init__(args[0], desc_len, *args[1:], **kwargs)

        self.sqrt_prior = self.torch.Tensor(self.sqrt_prior).to(self.torch_device)
        if self._lowmem:
            for key in ["energy", "force", "stress"]:
                self.likelihood[key] = self.torch.Tensor(self.likelihood[key]).to(self.torch_device)

    @abstractmethod
    def _get_descriptor_length(self):
        '''
        Gets & returns the length of the descriptor vector for this kind of MLIP model
        '''
        pass

    def _take_derivative_scalar(self, val, x):
        '''
        Take the derivative of the scalar val w.r.t x
        '''
        return self.torch.autograd.grad(outputs=[val], inputs=[x], grad_outputs=[self.torch.ones_like(val)], allow_unused=True, retain_graph=True)[0]
    
    def _take_derivative_vector(self, val, x):
        '''
        Take the derivative of scalar val w.r.t x by looping over x
        
        '''
        N = val.size()
        jac = self.torch.zeros(N[0], *x.shape).to(self.torch_device)
        for i in range(N[0]):
            v = val[i]
            jac[i, :, :] = self._take_derivative_scalar(v, x)
        return jac
    
    def get_property(self, name, atoms=None, allow_calculation=True):
        '''
        Overload of Calculator.get_property, converts from torch tensors to numpy arrays
        Allows for torch tensors to be stored in self.results between calls to 
        self.calculate
        
        '''
        return super().get_property(name, atoms, allow_calculation).detach().cpu().numpy()

    def resample_committee(self, committee_size=None):
        '''
        Resample the committee, based on the states of self.likelihood and self.sqrt_prior
        Populates self.committee_weights based on the newly sampled committee

        Parameters
        ----------
        committee_size : int, optional
            New size of the committee, if supplied.
            By default, a committee of size self.n_comm is drawn

        '''
        self._MPI_receive_all_selections() # Sync up with selections from other processes
        
        if committee_size is not None:
            self.n_comm = committee_size

        if self._lowmem:
            reg = (self.regularisation * self.torch.eye(self.n_desc)).to(self.torch_device)

            L_likelihood = self.torch.linalg.cholesky(sum([self.likelihood[key] for key in ["energy", "forces", "stress"]]) 
                                                 + reg)

            sqrt_posterior = L_likelihood + np.sqrt(self.prior_weight) * self.sqrt_prior

            Q, R = self.torch.linalg.qr(sqrt_posterior)
        
        else:
            l_list = []

            for key in ["energy", "forces", "stress"]:
                l_key = self.likelihood[key]
                if len(l_key):
                    l_list.extend(l_key)
            
            sqrt_posterior = self.torch.vstack(l_list + [self.sqrt_prior])
            Q, R = self.torch.linalg.qr(sqrt_posterior)

        
        z = self.torch.Tensor(self.rng.normal(loc=0, scale=1, size=(self.n_desc, self.n_comm))).to(self.torch_device)

        self.committee_weights = self.torch.linalg.solve_triangular(R, z, upper=True).T # zero mean committee, so no mean term

class TorchHALBiasPotential(TorchCommitteeCalculator, metaclass=ABCMeta):
    def _bias_energy(self, comm_energy):
        return self.torch.std(comm_energy)

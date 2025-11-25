from ase.atoms import Atoms
from .base_committee_calculator import BaseCommitteeCalculator, Calculator
import numpy as np
from abc import ABCMeta, abstractmethod
try:
    import torch
    from torch.autograd.functional import jacobian
    from mace.modules.utils import prepare_graph, get_edge_vectors_and_lengths, extract_invariant
    from e3nn import o3

    has_torch = True # True only if MACE/Torch optional deps are satisfied
except ImportError:
    has_torch = False

class BaseMACECalculator(BaseCommitteeCalculator, metaclass=ABCMeta):
    implemented_properties = ['energy', 'forces', 'stress', 'desc_energy', 'desc_forces', 'desc_stress', 
                              'comm_energy', 'comm_forces', 'comm_stress', 'bias_energy', 'bias_forces', 'bias_stress']
    def __init__(self, mace_calculator, committee_size, prior_weight,
                 num_layers=-1, invariants_only=True, **kwargs):
        '''
        
        Parameters
        ----------
        mace_calculator: mace.calculators.MACECalculator object
            MACE architecture to use to define a MACE descriptor
        committee_size: int
            Number of members in the linear committee
        prior_weight: float
            Weight corresponding to the prior matrix in the linear system
        num_layers: int (default: -1)
            Number of layers in the MACE model to keep for descriptor evaluation
            Default of -1 uses all but the readout layer (equivalent to MACECalculator.get_descriptors() default)
        invariants_only: bool
            Whether to only keep the invariants partition of the descriptor vector, see MACECalculator.get_descriptors
            for more details
        **kwargs: Keyword Args
            Extra keywork arguments fed to ase_uhal.BaseCommitteeCalculator
        '''

        assert has_torch, "PyTorch is required for MACE committees"

        self.mace_calc = mace_calculator

        self.model = mace_calculator.models[0]

        self.num_layers = num_layers
        self.invariants_only = invariants_only

        self.torch_device = self.model.atomic_numbers.get_device()

        if self.torch_device < 0:
            self.torch_device = "cpu"

        num_interactions = int(self.model.num_interactions)

        irreps_out = o3.Irreps(str(self.model.products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
        per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
        per_layer_features[-1] = (
            num_invariant_features  # Equivariant features not created for the last layer
        )

        if num_layers == -1:
                num_layers = num_interactions
        to_keep = np.sum(per_layer_features[:num_layers])

        self.l_max = l_max
        self.num_invariant_features = num_invariant_features
        self.num_layers = num_layers
        self.to_keep = to_keep

        # Build an atoms object with a species which the model can handle
        ats = Atoms(numbers=[self.model.atomic_numbers.detach().cpu().numpy()[0]], positions=[[0, 0, 0]])

        descriptor_size = self._descriptor_base(*self._prep_atoms(ats)).shape[0]

        super().__init__(committee_size, descriptor_size, prior_weight, **kwargs)
        
        self.sqrt_prior = torch.Tensor(self.sqrt_prior).to(self.torch_device)
        if self._lowmem:
            for key in ["energy", "force", "stress"]:
                self.likelihood[key] = torch.Tensor(self.likelihood[key]).to(self.torch_device)
        
    def _prep_atoms(self, atoms):
        '''
        Convert ASE atoms object into a format suitable for MACE models
        
        '''

        batch = self.mace_calc._atoms_to_batch(atoms).to_dict()

        ctx = prepare_graph(batch, compute_stress=True)

        return ctx.positions, batch["node_attrs"], batch["edge_index"], batch["shifts"], ctx.displacement

    def _descriptor_base(self, positions, attrs, edge_index, shifts, displacement):
        '''
        Base MACE descriptor, based on results from self._prep_atoms
        '''

        vectors, lengths = get_edge_vectors_and_lengths(
                    positions=positions,
                    edge_index=edge_index,
                    shifts=shifts
                )

        node_feats = self.model.node_embedding(attrs)
        edge_attrs = self.model.spherical_harmonics(vectors)
        edge_feats, cutoff = self.model.radial_embedding(
            lengths, attrs, edge_index, self.model.atomic_numbers
        )

        feats = []

        for i, (interaction, product) in enumerate(
                    zip(self.model.interactions, self.model.products)
                ):
            node_attrs_slice = attrs
            node_feats, sc = interaction(
                        node_attrs=node_attrs_slice,
                        node_feats=node_feats,
                        edge_attrs=edge_attrs,
                        edge_feats=edge_feats,
                        edge_index=edge_index,
                        cutoff=cutoff,
                        first_layer=(i == 0),
                    )
            node_feats = product(
                        node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
                    )
            
            feats.append(node_feats)

        node_feats_out = torch.cat(feats, dim=-1)

        if self.invariants_only:
                node_feats_out = extract_invariant(
                    node_feats_out,
                    num_layers=self.num_layers,
                    num_features=self.num_invariant_features,
                    l_max=self.l_max,
                )


        return node_feats_out[:, :self.to_keep].sum(dim=0)

    def _take_derivative_scalar(self, val, x):
        '''
        Take the derivative of the scalar val w.r.t x
        '''
        return torch.autograd.grad(outputs=[val], inputs=[x], grad_outputs=[torch.ones_like(val)], allow_unused=True, retain_graph=True)[0]
    
    def _take_derivative_vector(self, val, x):
        '''
        Take the derivative of scalar val w.r.t x by looping over x
        
        '''
        N = val.size()
        jac = torch.zeros(N[0], *x.shape).to(self.torch_device)
        for i in range(N[0]):
            v = val[i]
            jac[i, :, :] = self._take_derivative_scalar(v, x)
        return jac
    
    @abstractmethod
    def _bias_energy(self, comm_energy):
        pass
    
    def calculate(self, atoms, properties, system_changes):
        '''
        Calculation for descriptor properties, committee properties, normal properties, and HAL properties

        Descriptor properties use a "desc_" prefix, committee properties use "comm_", HAL (bias) properties use "hal_".
        
        '''
        super().calculate(atoms, properties, system_changes)

        struct = self._prep_atoms(atoms)

        positions = struct[0]
        displacement = struct[4]

        volume = atoms.get_volume()
                
        ### Energies
        # Always calculate, as we can use torch.autodiff later
        self.results["desc_energy"] = self._descriptor_base(*struct)

        self.results["comm_energy"] = self.committee_weights @ self.results["desc_energy"]
        
        if "energy" in properties or "forces" in properties or "stress" in properties:
            self.results["energy"] = torch.mean(self.results["comm_energy"])

        
        if "bias_energy" in properties or "bias_forces" in properties or "bias_stress" in properties:   
            self.results["bias_energy"] = self._bias_energy(self.results["comm_energy"])

            if "bias_forces" in properties:
                self.results["bias_forces"] = - self._take_derivative_scalar(self.results["bias_energy"], positions)
            if "bias_stress" in properties:
                self.results["bias_stress"] = - self._take_derivative_scalar(self.results["bias_energy"], displacement)[0, :, :] / volume



        ### Forces
        # Try to use previous results to save on recalculating autodiff gradients
        if "desc_forces" in properties:
            self.results["desc_forces"] = self._take_derivative_vector(self.results["desc_energy"], positions)

        if "comm_forces" in properties:
            if "desc_forces" in self.results.keys():
                F_comm = torch.tensordot(self.committee_weights, self.results["desc_forces"], dims=([1], [0]))
            else:
                F_comm = self._take_derivative_vector(self.results["comm_energy"], positions)

            self.results["comm_forces"] = F_comm

        if "forces" in properties:
            if "comm_forces" in self.results.keys():
                F = torch.mean(self.results["comm_forces"], dim=0)
            else:
                F = self._take_derivative_scalar(self.results["energy"], positions)

            self.results["forces"] = F

        ### Stresses
        # Achieved similar to forces
        if "desc_stress" in properties:
            self.results["desc_stress"] = self._take_derivative_vector(self.results["desc_energy"], displacement)[:, 0, :, :] / volume

        if "comm_stress" in properties:
            if "desc_stress" in self.results.keys():
                S_comm = torch.tensordot(self.committee_weights, self.results["desc_stress"], dims=([1], [0]))
            else:
                S_comm = self._take_derivative_vector(self.results["comm_stress"], positions) / volume

            self.results["comm_stress"] = S_comm

        if "stress" in properties:
            if "comm_stress" in self.results.keys():
                S = torch.mean(self.results["comm_stress"], dim=0)
            else:
                S = self._take_derivative_scalar(self.results["energy"], displacement)[0, :, :] / volume

            self.results["stress"] = S
    
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
            reg = (self.regularisation * torch.eye(self.n_desc)).to(self.torch_device)

            L_likelihood = torch.linalg.cholesky(sum([self.likelihood[key] for key in ["energy", "forces", "stress"]]) 
                                                 + reg)

            sqrt_posterior = L_likelihood + np.sqrt(self.prior_weight) * self.sqrt_prior

            Q, R = torch.linalg.qr(sqrt_posterior)
        
        else:
            l_list = []

            for key in ["energy", "forces", "stress"]:
                l_key = self.likelihood[key]
                if len(l_key):
                    l_list.extend(l_key)
            
            sqrt_posterior = torch.vstack(l_list + [self.sqrt_prior])
            Q, R = torch.linalg.qr(sqrt_posterior)

        
        z = torch.Tensor(self.rng.normal(loc=0, scale=1, size=(self.n_desc, self.n_comm))).to(self.torch_device)

        self.committee_weights = torch.linalg.solve_triangular(R, z, upper=True).T # zero mean committee, so no mean term


class MACEHALCalculator(BaseMACECalculator):
    name = "MACEHALCalculator"
    def _bias_energy(self, comm_energy):
        return torch.std(comm_energy)

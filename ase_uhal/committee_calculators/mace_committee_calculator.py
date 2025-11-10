from ase.atoms import Atoms
from .base_committee_calculator import BaseCommitteeCalculator, Calculator
import numpy as np

try:
    import torch
    from torch.autograd.functional import jacobian
    from mace.modules.utils import prepare_graph, get_edge_vectors_and_lengths, extract_invariant
    from e3nn import o3

    has_torch = True # True only if MACE/Torch optional deps are satisfied
except ImportError:
    has_torch = False


class MACECommitteeCalculator(BaseCommitteeCalculator):
    implemented_properties = ['forces', 'energy', 'free_energy']
    name = "MACECommitteeCalculator"
    def __init__(self, mace_calculator, committee_size, prior_weight, energy_weight=None, forces_weight=None, 
                 sqrt_prior=None, lowmem=False, random_seed=None, num_layers=-1, invariants_only=True, regularisation=1e-4, 
                 mpi_comm=None, torch__chunksize=100, **kwargs):

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

        descriptor_size = self.get_descriptor_energy(ats).shape[0]

        super().__init__(committee_size, descriptor_size, prior_weight, energy_weight, forces_weight, None, # Stress weight
                 sqrt_prior, lowmem, random_seed, regularisation, mpi_comm, **kwargs)
        
        self.sqrt_prior = torch.Tensor(self.sqrt_prior).to(self.torch_device)
        if self._lowmem:
            for key in ["energy", "force", "stress"]:
                self.likelihood[key] = torch.Tensor(self.likelihood[key]).to(self.torch_device)

        #self._desc_force = torch.func.jacrev(self._descriptor_base, 0, chunk_size=torch__chunksize)
        #self._comm_force = torch.func.jacrev(self._committee_energies, 0, chunk_size=torch__chunksize)

        # Use slower for loop manual jacobian to avoid torch Storage error
        self._desc_force = self.__manual_jac(self._descriptor_base)
        self._comm_force = self.__manual_jac(self._committee_energies)
        
    def _prep_atoms(self, atoms):

        batch = self.mace_calc._atoms_to_batch(atoms).to_dict()

        ctx = prepare_graph(batch)

        return ctx.positions, batch["node_attrs"], batch["edge_index"], batch["shifts"]

    def _descriptor_base(self, positions, attrs, edge_index, shifts):

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

    def __manual_jac(self, f):

        def _call(*args):

            def single_grad(args, i):
                desc_elem = f(*args)[i]
                grad_outputs = [torch.ones_like(desc_elem)]

                grad = torch.autograd.grad(outputs=[desc_elem], inputs=[args[0]], grad_outputs=grad_outputs, allow_unused=True)[0]

                return grad

            desc_len = self.n_desc

            jac = torch.zeros(desc_len, *args[0].shape).to(self.torch_device)

            for i in range(desc_len):
                jac[i, :, :] = single_grad(args, i)

            return jac
        return _call
    
    def _committee_energies(self, positions, attrs, edge_index, shifts):
        d = self._descriptor_base(positions, attrs, edge_index, shifts)
        return self.committee_weights @ d
    
    def _hal_energy(self, positions, attrs, edge_index, shifts):
        return torch.std(self._committee_energies(positions, attrs, edge_index, shifts))

    def get_descriptor_energy(self, atoms):
         return self._descriptor_base(*self._prep_atoms(atoms))
    
    def get_descriptor_force(self, atoms):
         # Get the jacobian w.r.t the first argument of self._descriptor_base (i.e. positions)
         #jacobian = torch.func.jacfwd(self._descriptor_base, 0)
         
         desc_force = self._desc_force(*self._prep_atoms(atoms))
         
         return desc_force
    
    def get_descriptor_stress(self, atoms):
         return super().get_descriptor_stress(atoms)
    
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

            L_likelihood = torch.linalg.cholesky(sum([self.likelihood[key] for key in ["energy", "force", "stress"]]) 
                                                 + reg)

            sqrt_posterior = L_likelihood + np.sqrt(self.prior_weight) * self.sqrt_prior

            Q, R = torch.linalg.qr(sqrt_posterior)
        
        else:
            l_list = []

            for key in ["energy", "force", "stress"]:
                l_key = self.likelihood[key]
                if len(l_key):
                    l_list.extend(l_key)
            
            sqrt_posterior = torch.vstack(l_list + [self.sqrt_prior])
            Q, R = torch.linalg.qr(sqrt_posterior)

        
        z = torch.Tensor(self.rng.normal(loc=0, scale=1, size=(self.n_desc, self.n_comm))).to(self.torch_device)

        self.committee_weights = torch.linalg.solve_triangular(R, z, upper=True).T # zero mean committee, so no mean term

    def get_committee_energies(self, atoms=None):
        '''
        Get energy predictions by each member of the committee

        Returns an array of length self.n_comm
        
        '''

        if atoms is None:
            atoms = self.atoms

        d = self.get_descriptor_energy(atoms)

        return (self._committee_energies(*self._prep_atoms(atoms))).detach().cpu().numpy()
    
    def get_committee_forces(self, atoms=None):
        '''
        Get force predictions by each member of the committee

        Returns an array of shape (self.n_comm, Nats, 3)
        
        '''

        if atoms is None:
            atoms = self.atoms


        return (self._comm_force(*self._prep_atoms(atoms))).detach().cpu().numpy()
    
    def get_committee_stresses(self, atoms=None):
        '''
        Get stress predictions by each member of the committee

        Returns an array of shape (self.n_comm, 9)

        
        '''

        if atoms is None:
            atoms = self.atoms
            
        d = self.get_descriptor_stress(atoms)

        return (self.committee_weights @ d).detach().cpu().numpy()

    def hal_calculate(self, atoms, properties, system_changes):
        '''
        Calculate the energy, force, and stress properties from the committee used for HAL

        Energy is the std() of the committee energies
        Forces and stresses are weighted averages, using weights of the energy predictions
        Stresses
        
        '''
        Calculator.calculate(self, atoms, properties, system_changes)

        if atoms is None:
            atoms = self.atoms

        props = self._prep_atoms(atoms)

        E_hal = self._hal_energy(*props)

        self.results["hal_energy"] = E_hal.detach().cpu().numpy()

        if "forces" in properties:
            # derivative w.r.t positions
            grad_outputs = [torch.ones_like(E_hal)]
            Fs = torch.autograd.grad(outputs=[E_hal], inputs=[props[0]], grad_outputs=grad_outputs, allow_unused=True)[0]
            self.results["hal_forces"] = Fs.detach().cpu().numpy()

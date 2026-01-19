from ase.atoms import Atoms
from .torch_committee_calculator import TorchCommitteeCalculator
import numpy as np
from abc import ABCMeta, abstractmethod

class BaseMACECalculator(TorchCommitteeCalculator, metaclass=ABCMeta):
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
            Extra keywork arguments fed to :class:`~ase_uhal.committee_calculators.TorchCommitteeCalculator`
        '''

        from mace.modules.utils import prepare_graph, get_edge_vectors_and_lengths, extract_invariant
        from e3nn import o3

        self.prepare_graph = prepare_graph
        self.get_edge_vectors_and_lengths = get_edge_vectors_and_lengths
        self.extract_invariant = extract_invariant
        self.o3 = o3

        self.mace_calc = mace_calculator

        self.model = mace_calculator.models[0]

        self.num_layers = num_layers
        self.invariants_only = invariants_only

        self.torch_device = self.model.atomic_numbers.get_device()

        if self.torch_device < 0:
            self.torch_device = "cpu"

        num_interactions = int(self.model.num_interactions)

        irreps_out = self.o3.Irreps(str(self.model.products[0].linear.irreps_out))
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

        super().__init__(committee_size, prior_weight, **kwargs)
        
    def _get_descriptor_length(self):
        # Build an atoms object with a species which the model can handle
        ats = Atoms(numbers=[self.model.atomic_numbers.detach().cpu().numpy()[0]], positions=[[0, 0, 0]])

        return self._descriptor_base(*self._prep_atoms(ats)).shape[0]
    
    def _prep_atoms(self, atoms):
        '''
        Convert ASE atoms object into a format suitable for MACE models
        
        '''

        batch = self.mace_calc._atoms_to_batch(atoms).to_dict()

        ctx = self.prepare_graph(batch, compute_stress=True)

        return ctx.positions, ctx.displacement, batch["node_attrs"], batch["edge_index"], batch["unit_shifts"], batch["cell"]

    def _descriptor_base(self, positions, displacement, attrs, edge_index, unit_shifts, cell):
        '''
        Base MACE descriptor, based on results from self._prep_atoms
        '''
        symmetric_displacement = 0.5 * (
            displacement + displacement.transpose(-1, -2)
        )  # From https://github.com/mir-group/nequip

        positions = positions + self.torch.einsum(
            "be,bec->bc", positions, symmetric_displacement
        )

        cell = cell.view(-1, 3, 3)

        cell = cell + self.torch.matmul(cell, symmetric_displacement)

        shifts = self.torch.einsum(
            "be,bec->bc",
            unit_shifts,
            cell,
        )

        vectors, lengths = self.get_edge_vectors_and_lengths(
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

        node_feats_out = self.torch.cat(feats, dim=-1)

        if self.invariants_only:
                node_feats_out = self.extract_invariant(
                    node_feats_out,
                    num_layers=self.num_layers,
                    num_features=self.num_invariant_features,
                    l_max=self.l_max,
                )


        return node_feats_out[:, :self.to_keep].sum(dim=0)
    
    def _desc_energy(self, *args):
        return self._descriptor_base(*args)
    
    def _comm_energy(self, *args):

        return self.committee_weights @ self._descriptor_base(*args)

    def _energy(self, *args):
        comm_energy = self._comm_energy(*args)
        
        return self.torch.mean(comm_energy)

    @abstractmethod
    def _bias_energy(self, *args):
        pass
    
    def _desc_forces(self, *args):
        desc_energy = self._descriptor_base(*args)
        #return self.torch.func.jacfwd(self._descriptor_base, argnums=0)(*args)
        return self._take_derivative_vector(desc_energy, args[0])
    
    def _comm_forces(self, *args):
        '''
        Use Vector jacobian product to compute comm forces, to save memory overheads
        '''

        def f(positions):
            return self._descriptor_base(positions, *args[1:])
        
        _, comm_force_func = self.torch.func.vjp(f, args[0])

        def g(weights):
            return comm_force_func(weights)[0]
        
        return self.torch.vmap(g)(self.committee_weights)
    
    def _forces(self, *args):
        comm_forces = self._comm_forces(*args)

        return self.torch.mean(comm_forces, dim=0)
    
    def _bias_forces(self, *args):
        bias_energy = self._bias_energy(*args)
        return self._take_derivative_scalar(bias_energy, args[0])
        return self.torch.func.jacfwd(self._bias_energy, argnums=0)(*args)
    
    def _desc_stress(self, *args):
        desc_energy = self._descriptor_base(*args)
        return self._take_derivative_vector(desc_energy, args[1])
        return self.torch.func.jacfwd(self._descriptor_base, argnums=1)(*args)
    
    def _comm_stress(self, *args):
        def f(displacements):
            return self._descriptor_base(args[0], displacements, *args[2:])
        
        _, comm_stress_func = self.torch.func.vjp(f, args[1])

        def g(weights):
            return comm_stress_func(weights)[0]
        
        return self.torch.vmap(g)(self.committee_weights)
    
    def _stress(self, *args):
        comm_stress = self._comm_stress(*args)

        return self.torch.mean(comm_stress, dim=0)
    
    def _bias_stress(self, *args):
        bias_energy = self._bias_energy(*args)
        return self._take_derivative_scalar(bias_energy, args[1])
        return self.torch.func.jacfwd(self._bias_energy, argnums=1)(*args)


    def calculate(self, atoms, properties, system_changes):
        '''
        Calculation for descriptor properties, committee properties, normal properties, and HAL properties

        Descriptor properties use a "desc_" prefix, committee properties use "comm_", HAL (bias) properties use "hal_".
        
        '''
        super().calculate(atoms, properties, system_changes)

        volume = atoms.get_volume()
        struct = self._prep_atoms(atoms)

        ### Energy
        if "desc_energy" in properties:
            self.results["desc_energy"] = self._descriptor_base(*struct)

        if "comm_energy" in properties:
            self.results["comm_energy"] = self._comm_energy(*struct)
        
        if "energy" in properties:
            self.results["energy"] = self._energy(*struct)

        if "bias_energy" in properties:
            self.results["bias_energy"] = self._bias_energy(*struct) 

        ### Forces
        if "desc_forces" in properties:
            self.results["desc_forces"] = -self._desc_forces(*struct)

        if "comm_forces" in properties:
            self.results["comm_forces"] = -self._comm_forces(*struct)

        if "forces" in properties:
            self.results["forces"] = -self._forces(*struct)
        
        if "bias_forces" in properties:
            self.results["bias_forces"] = -self._bias_forces(*struct)

        ### Stresses
        if "desc_stress" in properties:
            self.results["desc_stress"] = self._desc_stress(*struct)[:, 0, :, :] / volume

        if "comm_stress" in properties:
            self.results["comm_stress"] = self._comm_stress(*struct)[:, 0, :, :] / volume

        if "stress" in properties:
            self.results["stress"] = self._stress(*struct)[0, :, :] / volume
        
        if "bias_stress" in properties:
            self.results["bias_stress"] = self._bias_stress(*struct)[0, :, :] / volume


class MACEHALCalculator(BaseMACECalculator):
    name = "MACEHALCalculator"

    def _bias_energy(self, *args):
        comm_energy = self._comm_energy(*args)

        return self.torch.std(comm_energy)
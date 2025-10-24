from .base_committee_calculator import BaseCommitteeCalculator
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
    def __init__(self, mace_calculator, committee_size, descriptor_size, prior_weight, energy_weight=None, forces_weight=None, stress_weight=None, 
                 sqrt_prior=None, lowmem=False, random_seed=None, num_layers=-1, invariants_only=True):
        super().__init__(committee_size, descriptor_size, prior_weight, energy_weight, forces_weight, stress_weight, 
                 sqrt_prior, lowmem, random_seed)
        

        assert has_torch, "PyTorch is required for MACE committees"

        self.mace_calc = mace_calculator

        self.model = mace_calculator.models[0]

        self.num_layers = num_layers
        self.invariants_only = invariants_only

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

    def get_descriptor_energy(self, atoms):
         return self._descriptor_base(*self._prep_atoms(atoms))
    
    def get_descriptor_force(self, atoms):
         # Get the jacobian w.r.t the first argument of self._descriptor_base (i.e. positions)
         jacobian = torch.func.jacfwd(self._descriptor_base, 0)
         return jacobian(*self._prep_atoms(atoms))
    
    def get_descriptor_stress(self, atoms):
         return super().get_descriptor_stress(atoms)
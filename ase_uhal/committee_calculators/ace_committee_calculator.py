from .base_committee_calculator import BaseCommitteeCalculator


class ACECommitteeCalculator(BaseCommitteeCalculator):
    def __init__(self, committee_size, descriptor_size, prior_weight, energy_weight=None, forces_weight=None, stress_weight=None, 
                 sqrt_prior=None, lowmem=False, random_seed=None):
        super().__init__(committee_size, descriptor_size, prior_weight, energy_weight, forces_weight, stress_weight, 
                 sqrt_prior, lowmem, random_seed)
        pass

    
from .base_committee_calculator import BaseCommitteeCalculator
import os
import numpy as np
from typing import NamedTuple

file_root = os.path.dirname(os.path.abspath(__file__))


class ace_hypers(NamedTuple):
    elements: str
    order: int
    totaldegree: int
    rcut: float


class ACECommitteeCalculator(BaseCommitteeCalculator):
    implemented_properties = ['energy', 'forces', 'stress', 'desc_energy', 'desc_forces', 'desc_stress', 
                              'comm_energy', 'comm_forces', 'comm_stress', 'hal_energy', 'hal_forces', 'hal_stress']
    def __init__(self, ace_params, committee_size, prior_weight, energy_weight=None, forces_weight=None, stress_weight=None, 
                 sqrt_prior=None, lowmem=False, random_seed=None):
            
        from julia import Main


        self.jl = Main
        # ACEpotentials, plus some utilities
        self.jl.include(file_root + "/../data/_ace_utils.jl")

        if type(ace_params) == str:
            # assume filename
            self.model = self.jl.load_ace_model(ace_params)
        else:
            # assume set of ace hyperparameters
            if type(ace_params) == list:
                elements, order, totaldegree, rcut = ace_params
            else: # Dict
                elements = ace_params["elements"]
                order = ace_params["order"]
                totaldegree = ace_params["totaldegree"]
                rcut = ace_params["rcut"]
            self.model = self.jl.model_from_params(elements, order, totaldegree, rcut)

        descriptor_size = self.jl.length_basis(self.model)

        super().__init__(committee_size, descriptor_size, prior_weight, energy_weight, forces_weight, stress_weight, 
                 sqrt_prior, lowmem, random_seed)
        
    @property
    def committee_weights(self):
        return self._committee_weights
    
    @committee_weights.setter
    def committee_weights(self, new_weights):
        self._committee_weights = new_weights
        if new_weights is not None:
            self.jl.set_committee_b(self.model, [new_weights[i, :] for i in range(self.n_comm)])

    def _prep_atoms(self, atoms):
        '''
        Convert from ase atoms into the AtomsBase AbstractSystem, using ASEconvert
        '''
        numbers = atoms.get_atomic_numbers()
        positions = atoms.positions
        cell = atoms.cell[:, :]
        pbc = atoms.pbc

        return self.jl.convert_ats(numbers, positions, cell, pbc)
    
    def calculate(self, atoms, properties, system_changes):
        '''
        Calculation for descriptor properties, committee properties, normal properties, and HAL properties

        Descriptor properties use a "desc_" prefix, committee properties use "comm_", HAL properties use "hal_".
        
        '''
        super().calculate(atoms, properties, system_changes)
        all_props = [item for item in properties]

        for prop in properties:
            if "hal_" in prop: 
                if "comm_" + prop.split("_")[1] not in properties:
                    # HAL versions of properties require committee properties
                    all_props.append("comm_" + prop)

        if "desc_energy" not in self.results.keys():
            E, F, V = self.jl.eval_basis(self._prep_atoms(atoms), self.model)

            E = np.array(E); F = np.array(F); V = np.array(V)

            self.results["desc_energy"] = np.array(E)
            self.results["desc_forces"] = np.array(F)
            self.results["desc_stress"] = np.array(V) / atoms.get_volume()


        for key in ["energy", "forces", "stress"]:
            if key in all_props or "comm_" + key in all_props or "hal_" + key in all_props:
                comm_prop = self.committee_weights @ self.results["desc_" + key]
                self.results["comm_" + key] = comm_prop

                self.results[key] = np.mean(comm_prop, axis=0)

        if "hal_energy" in all_props:
            self.results["hal_energy"] = np.std(self.results["comm_energy"], axis=0)
        
        if "hal_force" in all_props or "hal_stress" in all_props:
            Es = self.results["comm_energy"] - self.results["energy"]
            Fs = self.results["comm_forces"] - self.results["forces"]
            Ss = self.results["comm_stress"] - self.results["stress"]

            if "hal_force" in all_props:
                self.results["hal_forces"] = np.mean([E * F for E, F in zip(Es, Fs)], axis=0)

            if "hal_stress" in all_props:
                self.results["hal_stresses"] = np.mean([E * S for E, S in zip(Es, Ss)], axis=0)
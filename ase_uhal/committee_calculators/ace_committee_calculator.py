from .base_committee_calculator import BaseCommitteeCalculator, HALBiasPotential
import os
import numpy as np
from abc import ABCMeta, abstractmethod

file_root = os.path.dirname(os.path.abspath(__file__))

class BaseACECalculator(BaseCommitteeCalculator, metaclass=ABCMeta):
    implemented_properties = ['energy', 'forces', 'stress', 'desc_energy', 'desc_forces', 'desc_stress', 
                              'comm_energy', 'comm_forces', 'comm_stress', 'bias_energy', 'bias_forces', 'bias_stress']
    def __init__(self, ace_params, committee_size, prior_weight, **kwargs):
        '''
        Parameters
        ----------
        ace_params: string or dict
            If ace_params is a string: Use ACEpotentials.load_model to load the model json file given by ace_params
            If ace_params is a dict: interpret ace_params as the hyperparameters dict for ACEpotentials.ace1_model e.g.

            .. code-block:: python
            
                ace_params = {"elements" : ["Si"], # Expects list of str 
                              "order" : 3, # Expects an int
                              "totaldegree" : 10, # Expects an int
                              "rcut" : 5.0 # Expects a float
                             }


        committee_size: int
            Number of members in the linear committee
        prior_weight: float
            Weight corresponding to the prior matrix in the linear system
        **kwargs: Keyword Args
            Extra keywork arguments fed to :class:`~ase_uhal.committee_calculators.BaseCommitteeCalculator`
        
        '''
            
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

        super().__init__(committee_size, descriptor_size, prior_weight, **kwargs)
        
    def _prep_atoms(self, atoms):
        '''
        Convert from ase atoms into the AtomsBase AbstractSystem, using ASEconvert
        '''
        numbers = atoms.get_atomic_numbers()
        positions = atoms.positions
        cell = atoms.cell[:, :]
        pbc = atoms.pbc

        return self.jl.convert_ats(numbers, positions, cell, pbc)
    
    @abstractmethod
    def _bias_energy(self, comm_energy):
        pass

    @abstractmethod
    def _bias_forces(self, comm_forces, comm_energy):
        pass

    @abstractmethod
    def _bias_stress(self, comm_stress, comm_energy):
        pass
    
    def calculate(self, atoms, properties, system_changes):
        '''
        Calculation for descriptor properties, committee properties, normal properties, and HAL properties

        Descriptor properties use a "desc_" prefix, committee properties use "comm_", HAL properties use "hal_".
        
        '''
        super().calculate(atoms, properties, system_changes)

        if "desc_energy" not in self.results.keys():
            # System has changed, need to recalculate base descriptors
            E, F, V = self.jl.eval_basis(self._prep_atoms(atoms), self.model)

            E = np.array(E); F = np.array(F).reshape(self.n_desc, -1, 3); V = np.array(V)

            self.results["desc_energy"] = np.array(E)
            self.results["desc_forces"] = np.array(F)
            self.results["desc_stress"] = np.array(V) / atoms.get_volume()

        for key in ["energy", "forces", "stress"]:
            if "comm_" + key in properties or key in properties or "bias_" + key in properties or key == "energy": 
                # Always calculate energy properties, as committee energies 
                # needed for force and stress bias calc
                comm_prop = np.tensordot(self.committee_weights, self.results["desc_" + key], axes=1)
                self.results["comm_" + key] = comm_prop

                self.results[key] = np.mean(comm_prop, axis=0)

        if "bias_energy" in properties:
            self.results["bias_energy"] = self._bias_energy(self.results["comm_energy"])

        if "bias_forces" in properties:
            self.results["bias_forces"] = self._bias_forces(self.results["comm_forces"], self.results["comm_energy"])

        if "bias_stress" in properties:
            self.results["bias_stress"] = self._bias_stress(self.results["comm_stress"], self.results["comm_energy"])

class ACEHALCalculator(HALBiasPotential, BaseACECalculator):
    name = "ACEHALCalculator"
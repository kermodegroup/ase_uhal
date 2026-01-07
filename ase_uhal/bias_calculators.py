'''
ase-compatible HAL-style bias calculator, using the committee error as an energy bias.

'''
from ase.calculators.calculator import Calculator
import numpy as np
from .committee_calculators.base_committee_calculator import BaseCommitteeCalculator
from abc import ABCMeta, abstractmethod
from ase.stress import voigt_6_to_full_3x3_stress

class BaseBiasCalculator(Calculator, metaclass=ABCMeta):
    '''
    ASE-compatible Bias calculator with adaptive biasing

    Derived from a combination of ACEHAL.bias_calc.BiasCalculator and ACEHAL.bias_calc.TauRelController from ACEHAL
    https://github.com/ACEsuit/ACEHAL/blob/main/ACEHAL/bias_calc.py
    
    '''
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters = {}
    name = 'HALCalculator'

    def __init__(self, mean_calc, committee_calc:BaseCommitteeCalculator, adaptive_tau=False, tau_rel=0.1, tau_hist=10, tau_delay=None, eps=0.2):
        '''
        '''

        super().__init__()
        self.mean_calc = mean_calc
        self.committee_calc = committee_calc
        self.adapt_tau = adaptive_tau
        self.tau_rel = tau_rel
        self.tau_hist = tau_hist
        if tau_delay is not None:
            self.tau_delay = tau_delay
        else:
            self.tau_delay = tau_hist
        self.tau = 0 # Default to no mixing, until specified by user, or changed by adaptive mode
        self.Fmean = None
        self.Fbias = None
        self.eps = eps

        # Validation
        assert self.tau_rel > 0
        assert self.mixing > 0
        assert self.tau_delay > 0

    @property
    def mixing(self):
        return 1/self.tau_hist
    
    @mixing.setter
    def mixing(self, mixing):
        self.tau_hist = 1/mixing
            
    def calculate(self, atoms, properties, system_changes):
        '''
        Overload of the BaseCalculator.calculate() abstract method
        Calculate props combining the results from the mean_calc with the results from the committee
        '''
        super().calculate(atoms, properties, system_changes)

        if len(system_changes):
            self.results = {}

        assert self.tau is not None

        self.committee_calc.calculate(atoms, ["bias_" + prop for prop in properties], system_changes)

        self.mean_calc.calculate(atoms, properties, system_changes)

        for prop in self.implemented_properties:
            if prop in properties:
                mean_prop = self.mean_calc.results[prop]
                if prop == "stress" and mean_prop.shape != (3, 3):
                    mean_prop = voigt_6_to_full_3x3_stress(mean_prop)
                # Use get_property as an interface to committee_calc, as torch-based comm calcs will use torch.Tensor in self.results 
                self.results[prop] = mean_prop - self.tau * self.committee_calc.get_property("bias_" + prop, atoms)

    def update_tau(self, atoms=None):
        '''
        Exponential mixing of mean force magnitude and mean biasing force magnitude 
        (without the tau biasing constant)

        Updates self.Fmean and self.Fbias based on self.mixing to mix between old values 
        and the new values provided as arguments 
        Modifies self.tau if self.tau_delay < 0, otherwise decreases self.tau_delay by 1
        '''
        Fmean = np.mean(np.linalg.norm(self.get_property("forces", atoms), axis=-1))
        Fbias = np.mean(np.linalg.norm(self.committee_calc.get_property("bias_forces", atoms), axis=-1))

        if self.Fmean is None or self.Fbias is None:
            self.Fmean = Fmean
            self.Fbias = Fbias
        else:
            self.Fmean = (1-self.mixing) * self.Fmean + self.mixing * Fmean
            self.Fbias = (1-self.mixing) * self.Fbias + self.mixing * Fbias
        
        if self.tau_delay > 0:
            # Positive delay means tau should not yet be updated
            # Updates to self.Fmean and self.Fbias needed to ensure 
            # smooth averaging when tau can be changed
            self.tau_delay -= 1
        else:
            self.tau = self.tau_rel * (self.Fmean / self.Fbias)

    @abstractmethod
    def get_score(self, atoms=None):
        pass

    def select_structure(self, ats):
        '''
        Alias for self.committee_calc.select_structure
        '''
        self.committee_calc.select_structure(ats)

    def resample_committee(self, committee_size=None):
        '''
        Alias for self.committee_calc.resample_committee
        '''
        self.committee_calc.resample_committee(committee_size)

    def sync(self):
        '''
        Alias of self.committee_calc.sync()
        '''
        self.committee_calc.sync()
    
class HALBiasCalculator(BaseBiasCalculator):
    '''
    Bias calculator with HAL scoring metric    
    '''
    def get_score(self, atoms=None):
        Fmean = np.linalg.norm(self.get_property("forces", atoms), axis=-1)
        Fbias = np.linalg.norm(self.committee_calc.get_property("bias_forces", atoms), axis=-1)
        
        F = Fbias / (Fmean + self.eps)

        s = np.exp(F) / np.sum(np.exp(F)) # Apply softmax

        return np.max(s)
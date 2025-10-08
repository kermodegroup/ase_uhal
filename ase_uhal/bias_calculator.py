'''
ase-compatible HAL-style bias calculator, using the committee error as an energy bias.

'''
from ase.calculators import BaseCalculator
from abc import ABCMeta, abstract_method
import numpy as np

class BiasCalculator(BaseCalculator, meta=ABCMeta):
    '''
    Abstract Base Class ASE calculator

    Derived from a combination of ACEHAL.bias_calc.BiasCalculator and ACEHAL.bias_calc.TauRelController from ACEHAL
    https://github.com/ACEsuit/ACEHAL/blob/main/ACEHAL/bias_calc.py
    
    '''
    implemented_properties = ['forces', 'energy', 'free_energy', 'stress']
    default_parameters = {}
    name = 'BiasCalculator'

    def __init__(self, mean_calc, adaptive_tau=False, tau_rel=0.1, tau_hist=10, tau_delay=10):
        '''
        '''
        self.mean_calc = mean_calc
        self.adapt_tau = adaptive_tau
        self.tau_rel = tau_rel
        self.mixing = 1 / tau_hist
        self.tau_delay = tau_delay
        self.tau = None
        self.Fmean = None
        self.Fbias = None

        # Validation
        assert self.tau_rel > 0
        assert self.mixing > 0
        assert self.tau_delay > 0

    @abstractmethod
    def committee_calculate(self, atoms, properties, system_changes):
        '''
        Method for getting property calculations from each member of the committee
        Returns a dict with keys of Es, Fs, Vs for committee energies, forces, and stresses (if requested by properties)
        
        '''
        pass

    def update_tau(self, Fmean, Fbias):
        '''
        Exponential mixing of mean force magnitude and 
        '''
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
            
    def calculate(self, atoms, properties, system_changes):
        '''
        Overload of the BaseCalculator.calculate() abstract method
        Calculate props combining the results from the mean_calc with the results from the committee
        '''
        assert self.tau is not None

        committee_props = self.committee_calculate(atoms, properties, system_changes)

        self.mean_calc.calculate(atoms, properties, system_changes)

        for prop in self.implemented_properties:
            if prop in properties:
                self.results[prop] = self.mean_calc.results[prop] - self.tau * committee_props[prop]
        
        if self.adapt_tau:
            fmean = np.mean(np.linalg.norm(self.mean_calc.properties["forces"], axis=-1))
            fbias = np.mean(np.linalg.norm(committee_props["forces"], axis=-1))
            self.update_tau(fmean, fbias)
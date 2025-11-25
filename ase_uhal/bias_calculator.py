'''
ase-compatible HAL-style bias calculator, using the committee error as an energy bias.

'''
from ase.calculators.calculator import Calculator
import numpy as np
from .committee_calculators.base_committee_calculator import BaseCommitteeCalculator


class BiasCalculator(Calculator):
    '''
    ASE-compatible Bias calculator with adaptive biasing

    Derived from a combination of ACEHAL.bias_calc.BiasCalculator and ACEHAL.bias_calc.TauRelController from ACEHAL
    https://github.com/ACEsuit/ACEHAL/blob/main/ACEHAL/bias_calc.py
    
    '''
    implemented_properties = ['forces', 'energy', 'stress']
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

        assert self.tau is not None

        self.committee_calc.calculate(atoms, ["bias_" + prop for prop in properties], system_changes)

        self.mean_calc.calculate(atoms, properties, system_changes)

        for prop in self.implemented_properties:
            if prop in properties:
                # Use get_property as an interface to committee_calc, as torch-based comm calcs will use torch.Tensor in self.results 
                self.results[prop] = self.mean_calc.results[prop] - self.tau * self.committee_calc.get_property("bias_" + prop, atoms)

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

    def get_hal_score(self, atoms=None):
        Fmean = np.linalg.norm(self.get_property("forces", atoms), axis=-1)
        Fbias = np.linalg.norm(self.committee_calc.get_property("bias_forces", atoms), axis=-1)
        
        F = Fbias / (Fmean + self.eps)

        s = np.exp(F) / np.sum(np.exp(F)) # Apply softmax

        return np.max(s)
    

class StructureSelector():
    '''
    Dynamics observer which decides whether to select a structure based on the score values of a trajectory

    Selects a structure if the score is above some threshold, and if the previous and next scores are both lower
    
    '''

    def __init__(self, bias_calc, threshold="adaptive", auto_resample=True, delay=10, mixing=0.1, thresh_mul=1.2):
        '''
        Parameters
        -----------
        bias_calc: ase_uhal.HalCalculator object
            HAL calculator used to run dynamics

        threshold: float or "adaptive" (default: "adaptive")
            threshold for determining when peaks in the score should be detected
            "adaptive" computes a new threshold after each call, by mixing scores from each call
            to the observer (see delay, mixing, and thresh_mul for more details)

        auto_resample: bool (default: False)
            Whether the biasing committee should be automatically resampled after every selection

        delay: int (default: 10)
            When the threshold is determined automatically, don't update it for this many calls (to allow
            mixing to settle on a reasonable value)
        
        mixing: float (default: 0.1)
            Mixing strength when threshold is automatically determined
            mixed_score = (1-mixing) * mixed_score + mixing * new_score

        thresh_mul: float (default: 1.2)
            Multiplier on the mixed score to determine the threshold
            threshold = thresh_mul * mixed_score 



        '''
        self.bias_calc = bias_calc

        self.delay = delay
        self.mixing = mixing
        self.thresh_mul = thresh_mul
        self.auto_resample = auto_resample

        if type(threshold) == str and threshold.lower() == "adaptive":
            self.adaptive = True
            self.reset_threshold()
        else:
            self.adaptive = False
            self.threshold = threshold
            assert self.threshold > 0

        self._prev_score = 0
        self._atoms = None
        self._peak = False

    def reset_threshold(self):
        '''
        Sets threshold to np.inf, resets the internal mixing state and the delay back to self.delay
        
        '''
        self._delay = self.delay

        self.threshold = np.inf

        self._mixed_score = None

    def _update_threshold(self, score):
        if self._mixed_score is None:
            self._mixed_score = score
        else:
            self._mixed_score = (1-self.mixing) * self._mixed_score + self.mixing * score

        if self._delay > 0:
            self._delay -= 1
        else:
            self.threshold = self.thresh_mul * self._mixed_score

    def __call__(self):
        score = self.bias_calc.get_hal_score()


        if score > self.threshold and score > self._prev_score:
            # Possibly at a peak above the threshold
            self._atoms = self.bias_calc.atoms.copy()
            self._peak = True
        elif self._peak:
            # Was at a peak previously, now select that previous structure

            self.bias_calc.committee_calc.select_structure(self._atoms)

            if self.auto_resample:
                self.bias_calc.committee_calc.resample_committee()

            self._peak = False
            self._atoms = None

        if self.adaptive:
            self._update_threshold(score)

        self._prev_score = score
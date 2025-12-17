import numpy as np


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
        score = self.bias_calc.get_score()


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
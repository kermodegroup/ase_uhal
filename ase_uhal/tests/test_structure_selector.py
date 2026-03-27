from mace.calculators import mace_mp
from ase.build import bulk
# ase_uhal imports
from ase_uhal.bias_calculators import HALBiasCalculator
from ase_uhal.committee_calculators import MACEHALCalculator
from ase_uhal import StructureSelector
from ase.md.langevin import Langevin
from ase.units import fs
import numpy as np

ref_ats = bulk("Si", cubic=True)

def test_structure_selector():
    '''
    Run MD with bias + selector active
    
    '''
    ats = ref_ats.copy()

    ### Setup ase_uhal classes
    mace_calc = mace_mp("medium-mpa-0", default_dtype="float64") # normal MACE MPA medium model calculator (from mace_torch)

    comm_calc = MACEHALCalculator(mace_calc, 
                                        committee_size=20,
                                        prior_weight=0.1,
                                        energy_weight=1, forces_weight=100,
                                        lowmem=False,
                                        batch_size=16,
                                        rng=np.random.RandomState(42))

    comm_calc.resample_committee() 

    hal_calc = HALBiasCalculator(mean_calc=mace_calc,
                            committee_calc=comm_calc,
                            adaptive_tau=True,
                            tau_rel=0.1,
                            tau_hist=10,
                            tau_delay=30)

    selector = StructureSelector(bias_calc=hal_calc,
                                threshold="adaptive",
                                auto_resample=True,
                                delay=10,
                                mixing=0.1,
                                thresh_mul=1.5)
    
    ats.calc = hal_calc

    dyn = Langevin(ats, 1*fs, temperature_K=300, friction=0.01 / fs)
    # Attach observers to dynamics, to be automatically called during the run
    dyn.attach(hal_calc.update_tau)
    dyn.attach(selector, 2)

    dyn.run(40)

    selector.reset_scoring()
    hal_calc.reset_tau()

    dyn.run(40)
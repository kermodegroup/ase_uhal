import mpi4py

from mace.calculators import mace_mp
from ase.build import bulk
from ase.atoms import Atoms
import matplotlib.pyplot as plt
import numpy as np
from ase.md.langevin import Langevin
from ase.units import fs
from ase.io import write
# ase_uhal imports
from ase_uhal.bias_calculators import HALBiasCalculator
from ase_uhal.committee_calculators import MACEHALCalculator
from ase_uhal import StructureSelector


### Setup ase_uhal classes
mace_calc = mace_mp() # normal MACE MPA medium model calculator (from mace_torch)

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



### Setup a cell of silicon bulk, with an interstitial metal impurity atom

# Metal impurity atoms
impurities = ["Fe", "Cu", "Ni", "Al"]

# Choose the impurity species based on MPI rank
my_rank = comm_calc.rank # Get the MPI rank directly from the calculator
my_impurity = impurities[my_rank]

Si = bulk("Si", cubic=True)

inter_pos = np.array([0.5, 0.25, 0.25]) * Si.cell[0, 0]

ats = Si * (2, 2, 2) + Atoms(my_impurity, positions=[inter_pos])

# Ranks can also use different observation weights
if my_rank == 0:
    comm_calc.energy_weight = 2
    comm_calc.forces_weight = 200

dyn = Langevin(ats, 1*fs, temperature_K=300, friction=0.01 / fs)
# Attach observers to dynamics, to be automatically called during the run
dyn.attach(hal_calc.update_tau)
dyn.attach(selector, 2)

### Run Dynamics
# Runs each MD on different interstitial species in parallel
# If any rank selects a structure, this is sent via MPI to all other ranks in the communicator
# At the time of the send, the energy, force, and virial weights are also sent
# to ensure that each rank maintains the same linear system
dyn.run(1000)

### Save down the selected structures
comm_calc.sync() # Make sure all MPI processes are fully synced, and all messages are properly recieved

if my_rank == 0:
    selected_structures = comm_calc.selected_structures

    write("Selected_Silicon_Impurities.xyz", selected_structures)
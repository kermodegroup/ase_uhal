from mace.calculators import mace_mp
from ase.build import bulk
import pytest
import numpy as np
# ase_uhal imports
from ase_uhal.committee_calculators import MACEHALCalculator
from ase_uhal.committee_calculators.base_committee_calculator import COMM_WORLD
from ase_uhal.distillation import distill_dataset

ref_ats = bulk("Si", cubic=True)

@pytest.mark.parametrize("comm", [None, COMM_WORLD])
def test_distillation(comm, allclose):
    '''
    Check that distillation prior matches internal linear system, both with and without MPI
    
    '''

    ### Setup ase_uhal classes
    mace_calc = mace_mp("medium-mpa-0", default_dtype="float64") # normal MACE MPA medium model calculator (from mace_torch)

    comm_calc = MACEHALCalculator(mace_calc, 
                                        committee_size=20,
                                        prior_weight=1,
                                        energy_weight=1, forces_weight=1,
                                        lowmem=False,
                                        batch_size=16,
                                        rng=np.random.RandomState(42),
                                        comm=comm)
    

    vol_range = np.linspace(0.8, 1.2, 7)

    dataset = []

    for vol in vol_range:
        cell = ref_ats.cell[:, :].copy()
        new_cell = cell * vol
        ats = ref_ats.copy()
        ats.set_cell(new_cell, scale_atoms=True)
        dataset.append(ats.copy())

    distilled = distill_dataset(dataset, comm_calc)

    for ats in dataset:
        comm_calc.select_structure(ats)

    # Construct same prior using torch
    l_list = []

    for key in ["energy", "forces", "stress"]:
        l_key = comm_calc.likelihood[key]
        if len(l_key):
            l_list.extend(l_key)
            
    sqrt_posterior = comm_calc.torch.vstack(l_list + [comm_calc.sqrt_prior])
    Q, R = comm_calc.torch.linalg.qr(sqrt_posterior)

    R = R.detach().numpy()

    assert allclose(R.T @ R, distilled.T @ distilled)
    


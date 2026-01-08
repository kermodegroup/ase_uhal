import ase_uhal
import pytest

from ase.build import bulk
from ase_uhal import committee_calculators as comm
import numpy as np

import ase_uhal.bias_calculators
import ase_uhal.committee_calculators

from .utils import finite_difference_forces, finite_difference_stress
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin
from ase.units import fs

ref_ats = bulk("Si", cubic=True)

try:
    from mace.calculators import mace_mp
    mpa = mace_mp("medium-mpa-0", default_dtype="float64")
except ImportError:
    mpa = None

bias_calcs = [ase_uhal.bias_calculators.HALBiasCalculator]

bias_calcs = {
    cls.__name__ : cls for cls in bias_calcs
}

comm_calc = ase_uhal.committee_calculators.MACEHALCalculator(mpa, 10, 0.1)

@pytest.mark.parametrize("calc_name", bias_calcs.keys())
class TestBiasCalcs():
    def set_up_calc(self, calc_name, required_properties=[]):
        if mpa is None:
            pytest.skip("mace-torch module is not installed")
        
        cls = bias_calcs[calc_name]

        for prop in required_properties:
            if prop not in cls.implemented_properties:
                pytest.skip(f"{cls.__name__} does not implement {prop}")
            if "bias_" + prop not in comm_calc.implemented_properties:
                pytest.skip(f"{comm_calc.__name__} does not implement bias_{prop}")

        calc = cls(mean_calc=mpa, committee_calc=comm_calc, adaptive_tau=True)
        calc.resample_committee() # Alias of calc.committee_calc.resample_committee()

        return calc
    
    def test_bias_forces(self, calc_name, allclose):
        # Test if bias forces are a derivative of the bias energy
        ats = ref_ats.copy()
        ats.rattle(1e-1, seed=42)

        calc = self.set_up_calc(calc_name, required_properties=["energy", "forces"])
        calc.tau = 0.2 # Need to manually set tau to enable biasing

        finite_difference_forces(calc, ats, allclose)

    
    def test_bias_stress(self, calc_name, allclose):
        # Test if bias stresses are a derivative of the bias energy
        ats = ref_ats.copy()
        ats.rattle(1e-1, seed=42)
        cell = ats.cell[:, :].copy()
        for i in range(3):
            cell[i, i] += 0.3
        ats.set_cell(cell, scale_atoms=True)

        calc = self.set_up_calc(calc_name, required_properties=["energy", "stress"])
        calc.tau = 0.2 # Need to manually set tau to enable biasing, as default is zero for adaptive_tau=True

        finite_difference_stress(calc, ats, allclose, dx=1e-10, atol=1e-2)

    def test_bias_dynamics(self, calc_name, allclose):
        ats = ref_ats.copy()
        calc = self.set_up_calc(calc_name, required_properties=["forces", "stress"])
        ats.calc = calc

        rng = np.random.RandomState(42)

        MaxwellBoltzmannDistribution(ats, temperature_K=300, rng=rng)

        dyn = Langevin(ats, 1*fs, temperature_K=300, rng=rng, friction=0.01/fs)
        dyn.attach(calc.update_tau)

        dyn.run(10*calc.tau_delay) # Run long enough for tau to be reasonably stable

        tau_rels = []

        for i in range(30):
            dyn.run(1)
            # Reverse engineer an approximate tau rel from the current tau + forces
            tau_rels.append(np.average(np.abs(calc.tau*calc.committee_calc.get_bias_forces()/calc.mean_calc.get_forces())))
        
        tau_rel = np.mean(tau_rels)

        # Check if the biasing strength means the bias forces are scaled correctly w.r.t. the mean forces
        # Relation is approximate, as tau_rel determined by mixing w/ previous states 
        assert np.abs(tau_rel - calc.tau_rel) < 5e-2

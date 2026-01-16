import ase_uhal
import pytest

from ase.build import bulk
from ase_uhal import committee_calculators as comm
import numpy as np
from .utils import finite_difference_forces, finite_difference_stress

ref_ats = bulk("Si", cubic=True)

try:
    import juliacall
    has_julia = True
except ImportError:
    has_julia = False

try:
    from mace.calculators import mace_mp
    mpa = mace_mp("medium-mpa-0", default_dtype="float64")
except ImportError:
    mpa = None

shared_params = {
    "committee_size" : 10,
    "prior_weight" : 0.1,
    "energy_weight" : 10,
    "forces_weight" : 10,
    "stress_weight" : 10
}


mace_params = shared_params.copy()
mace_params.update({
    "mace_calculator" : mpa,
})

ace_params = shared_params.copy()
ace_params.update({
    "ace_params" : {
        "elements" : ["Si"],
        "order" : 3,
        "totaldegree" : 10,
        "rcut" : 5.0 
    }
})

mace_calcs = [comm.MACEHALCalculator]
ace_calcs = [comm.ACEHALCalculator]

mace_data = {calc.__name__ : (calc, mace_params) for calc in mace_calcs}
ace_data = {calc.__name__ : (calc, ace_params) for calc in ace_calcs}

all_data = mace_data.copy()
all_data.update(ace_data)

@pytest.mark.parametrize("calc_name", all_data.keys())
class TestCommitteeCalcs():

    def set_up_calc(self, calc_name, required_properties=[]):
        if "MACE" in calc_name:
            if mpa is None:
                pytest.skip("mace-torch module is not installed")
        elif "ACE" in calc_name:
            if not has_julia:
                pytest.skip("Julia python module not installed")

        cls, params = all_data[calc_name]

        rng = np.random.RandomState(42)

        for prop in required_properties:
            if prop not in cls.implemented_properties:
                pytest.skip(f"{cls.__name__} does not implement {prop}")

        calc = cls(**params, rng=rng)
        calc.resample_committee()

        return calc
    
    @pytest.mark.parametrize("property", ["desc_forces", "comm_forces", "forces", "bias_forces"])
    def test_force_derivative(self, allclose, calc_name, property):
        # Compare direct force predictions to a finite differences scheme
        p = property.split("_")
        if len(p) > 1:
            # prop_forces
            prop = p[0] + "_"
        else:
            # forces
            prop = ""

        energy_prop = prop + "energy"
        force_prop = prop + "forces"

        ats = ref_ats.copy()
        ats.rattle(1e-1, seed=42)
        calc = self.set_up_calc(calc_name, required_properties=[energy_prop, force_prop])

        finite_difference_forces(calc, ats, allclose, energy_prop, force_prop, dx=1e-5)

    @pytest.mark.parametrize("property", ["desc_stress", "comm_stress", "stress", "bias_stress"])
    def test_stress_derivative(self, allclose, calc_name, property):
        # Compare direct force predictions to a finite differences scheme
        p = property.split("_")
        if len(p) > 1:
            # prop_forces
            prop = p[0] + "_"
        else:
            # forces
            prop = ""

        energy_prop = prop + "energy"
        stress_prop = prop + "stress"
        
        ats = ref_ats.copy()
        cell = ats.cell[:, :].copy()

        rng = np.random.RandomState(42)
        for i in range(3):
            for j in range(3):
                cell[i, j] += rng.rand() * 0.6 - 0.3
        ats.set_cell(cell, scale_atoms=True)
        ats.rattle(1e-1, seed=42)

        calc = self.set_up_calc(calc_name, required_properties=[energy_prop, stress_prop])

        finite_difference_stress(calc, ats, allclose, energy_prop, stress_prop, dx=1e-6)

    def test_committee_resample(self, calc_name):

        calc = self.set_up_calc(calc_name)

        if issubclass(calc.__class__, ase_uhal.committee_calculators.TorchCommitteeCalculator):
            def to_numpy(tensor):
                return tensor.detach().cpu().numpy()
        else:
            def to_numpy(arr):
                return arr

        calc.resample_committee(10)
        cw = to_numpy(calc.committee_weights)
        assert cw.shape[0] == 10

        calc.resample_committee(100)
        cw = to_numpy(calc.committee_weights)
        assert cw.shape[0] == 100


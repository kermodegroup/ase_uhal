import numpy as np


def finite_difference_forces(calc, atoms, allclose, energy_property="energy", force_property="forces", dx=1e-5, atol=1e-3):
    '''
    Calculate forces through finite differences of the property defined by energy property, and compare to the result of 
    force_property
    '''

    F_ref = calc.get_property(force_property, atoms)

    F_diff = np.zeros_like(F_ref)

    for i in range(len(atoms)):
        for j in range(3):
            ats = atoms.copy()
            pos = ats.positions.copy()

            pos[i, j] += dx
            ats.set_positions(pos)
            E1 = calc.get_property(energy_property, ats)


            pos[i, j] -= 2*dx
            ats.set_positions(pos)
            E2 = calc.get_property(energy_property, ats)

            F_diff[..., i, j] = (E2 - E1) / (2*dx)
    assert allclose(F_ref, F_diff, atol=atol)


def finite_difference_stress(calc, atoms, allclose, energy_property="energy", stress_property="stress", dx=1e-5, atol=1e-3):
    '''
    Calculate stresses through finite differences of the property defined by energy property, and compare to the result of 
    stress_property
    '''

    atoms.calc = calc

    S_ref = calc.get_property(stress_property, atoms)
    S_diff = np.zeros_like(S_ref)

    V = atoms.get_volume()
    for i in range(3):
        for j in range(3):
            ats = atoms.copy()
            cell = ats.cell[:, :].copy()

            eps = np.eye(3)

            eps[i, j] += dx
            new_cell = cell @ eps
            ats.set_cell(new_cell, scale_atoms=True)
            E1 = calc.get_property(energy_property, ats)

            eps[i, j] -= 2*dx
            new_cell = cell @ eps
            ats.set_cell(new_cell, scale_atoms=True)
            E2 = calc.get_property(energy_property, ats)

            S_diff[..., i, j] = -(E2 - E1) / (2*dx * V)
    assert allclose(S_ref, S_diff, atol=atol)
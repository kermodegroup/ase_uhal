import numpy as np
from ase.calculators.calculator import all_changes


def build_lin_systems(ds, weights, numbers, calc, compress_memory=False):
    '''
    Build 
    
    '''
    N_obs = np.sum(numbers, axis=0)

    energy_sys = None
    force_sys = None
    stress_sys = None
    # Build separate linear systems here, so we can potentially re-weight energies, forces, ... without needing to
    # recalculate descriptors
    if N_obs[0]:
        energy_sys = np.zeros((N_obs[0], calc.n_desc))

    if N_obs[1]:
        force_sys = np.zeros((N_obs[1], calc.n_desc))

    if N_obs[2]:
        stress_sys = np.zeros((N_obs[2], calc.n_desc))

    i_E = 0
    i_F = 0
    i_S = 0

    props = ["desc_energy", "desc_forces", "desc_stress"]

    for i, atoms in enumerate(ds):
        w = weights[i]
        n = numbers[i, :]

        properties = [props[j] for j in range(3) if w[j] is not None]
        # Calculate everything needed in a single pass, potentially saving some computational cost
        calc.calculate(atoms, properties, all_changes)

        if w[0] is not None:
            energy_sys[i_E, :] = calc.get_descriptor_energy(atoms) * w[0]
            i_E += 1
        
        if w[1] is not None:
            N_f = len(atoms) * 3
            force_sys[i_F:i_F + N_f, :] = calc.get_descriptor_forces(atoms) * w[1]
            i_F += N_f
        
        if w[2] is not None:
            stress_sys[i_S:i_S + 9, :] = calc.get_descriptor_stress(atoms) * w[2]

    if compress_memory:
        # Take QR decomposition from each system, to compress into a more memory-efficient (but less informative!) state
        systems = []
        for sys in [energy_sys, force_sys, stress_sys]:
            Q, R = np.linalg.qr(sys)
            systems.append(R)
        return systems
    else:
        return energy_sys, force_sys, stress_sys


def solve_lin_system(energy_sys=None, force_sys=None, stress_sys=None, prior=None):
    system = []

    for sys in [energy_sys, force_sys, stress_sys, prior]:
        if sys is not None:
            system.append(sys)

    assert len(system), "Linear system has no design matrix and prior!"

    sqrt_posterior = np.vstack(system)
    Q, R = np.linalg.qr(sqrt_posterior)

    return R



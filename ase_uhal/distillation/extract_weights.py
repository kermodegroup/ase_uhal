import numpy as np


def extract_weights_nobservations_single(atoms, weight_keys, default_weights):
    '''
    Find the total, energy, force, and stress weights either using the default_weights, or from keys in atoms.info

    Also compute the number of energy, force, and stress observations, based on whether the weight is none or not
    '''
    weights = [item for item in default_weights] # copy the default weights
    numbers = np.zeros(3)

    for i in range(3):
        if weight_keys[i] in atoms.info.keys():
            weight_keys[i] = atoms.info[weight_keys[i]]
            assert weight_keys[i] > 0

    if weights[0] is not None:
        # Total weight is not None, therefore compute some observations

        if weights[1] is not None:
            # Energy
            numbers[0] = 1
            weights[1] *= weights[0]

        if weights[2] is not None:
            # Forces
            numbers[1] = 3 * len(atoms)
            weights[2] *= weights[0]

        if weights[3] is not None:
            # Stress
            numbers[2] = 9
            weights[3] *= weights[0]
    else:
        weights = [None]*4

    return weights[1:], numbers

def extract_weights_nobservations(ds, weight_keys, default_weights):
    weights = []
    numbers = np.zeros((len(ds), 3))

    for i, atoms in enumerate(ds):
        w, n = extract_weights_nobservations_single(atoms, weight_keys, default_weights)
        weights.append(w)
        numbers[i, :] = n

    return weights, numbers

def estimate_memory_spike(numbers, n_desc, nproc):
    nsum = np.sum(numbers, axis=-1)
    ntot = np.sum(nsum) # Total number of observations in dataset
    nmax = np.max(nsum) # Max number of observations in one structure

    nlin = ntot * n_desc # Total number of elements in the linear system
    nobsmax = nmax * n_desc # Size of the largest chunk of the linear system

    float_in_GB = 8 / 10**9 # Size of a float in Gigabytes

    memest = (np.ceil(nlin/nproc) + nobsmax) * float_in_GB

    return memest
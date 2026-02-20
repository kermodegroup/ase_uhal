
from .extract_weights import extract_weights_nobservations
from .build_linear_system import build_lin_systems, solve_lin_system
import numpy as np

try:
    from mpi4py import MPI
    has_mpi = True

except ImportError:
    has_mpi = False


def _MPI_split_datasets(dataset, rank, num_procs):
    '''
    Split dataset across MPI ranks, if MPI is being used.
    '''
    if has_mpi:
        lens = [len(ats) for ats in dataset] # Size of each structure
        idxs = np.argsort(lens)
        sorted_dataset = [dataset[idx] for idx in idxs] # Sort dataset by size
        my_dataset = sorted_dataset[rank::num_procs]
        return my_dataset
    else:
        return dataset
    
def _MPI_gather_distillations(R, rank, target_rank, comm):
    '''
    Combine the results of distillation across different MPI ranks onto target_rank
    
    '''
    if comm is not None:
        R_list = comm.Gather(R, root=target_rank)
        if rank == target_rank:
            B = np.vstack(R_list)
            Q, R = np.linalg.qr(B)
            return R
        else:
            # Don't return anything which could be mistaken as the overall distilled result
            return None
    else:
        return R



def distill_dataset(dataset, calc, total_weight_key=None, energy_weight_key=None, forces_weight_key=None, 
                    stress_weight_key=None, sqrt_prior=None, compress_memory=False, MPI_target_rank=0):
    '''
    Given a dataset and a committee calculator, use the calculator descriptor to assemble a linear system from the dataset with given energy, force, stress weights.
    Then, solve for the posterior covariance of this system, which is equivalent to a new prior on any extensions to the dataset.

    For MPI usage, take all configuration from the committee calculator
    
    '''
    comm = calc.comm
    rank = calc.rank
    n_proc = calc.comm_size

    dataset = _MPI_split_datasets(dataset, rank, n_proc)

    weights, numbers = extract_weights_nobservations(dataset, [total_weight_key, energy_weight_key, forces_weight_key, stress_weight_key],
                                  [1.0, calc.energy_weight, calc.forces_weight, calc.stress_weight]) # Use 1.0 as the default total weight
    
    systems = build_lin_systems(dataset, weights, numbers, calc, compress_memory)

    if sqrt_prior is None:
        sqrt_prior = calc.sqrt_prior * calc.prior_weight

    R = solve_lin_system(*systems, sqrt_prior=sqrt_prior, num_mpi_procs=n_proc)

    del systems # Minimise memory overhead by removing the raw linear system before the MPI gather

    R = _MPI_gather_distillations(R, rank, target_rank=MPI_target_rank, comm=comm)

    return R
    
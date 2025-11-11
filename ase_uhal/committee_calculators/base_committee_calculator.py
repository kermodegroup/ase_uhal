from abc import ABCMeta, abstractmethod
from ase.calculators.calculator import Calculator
import numpy as np
from scipy.linalg import solve_triangular

try:
    from mpi4py import MPI
    has_mpi = True

    COMM_WORLD = MPI.COMM_WORLD

except ImportError:
    has_mpi = False
    COMM_WORLD = None


class BaseCommitteeCalculator(Calculator, metaclass=ABCMeta):
    implemented_properties = ['forces', 'energy', 'free_energy', 'stress']
    default_parameters = {}
    name = 'BaseCommitteeCalculator'

    def __init__(self, committee_size, descriptor_size, prior_weight, energy_weight=None, forces_weight=None, stress_weight=None, 
                 sqrt_prior=None, lowmem=False, random_seed=None, regularisation=1e-4, mpi_comm=COMM_WORLD, **kwargs):
        '''
        Parameters
        ----------
        committee_size : int
            size of the committee used to approximate the posterior
        
        descriptor_size : int
            length of the descriptor vector. Should be set by child classes, rather than the end user
        
        prior_weight: float
            Weight given to the prior, relative to the likelihood, in forming the posterior
        
        energy_weight: float or None, optional
            Weight given to energy observations in forming the likelihood. If None, energy observations will be ommitted
        
        forces_weight: float or None, optional
            Weight given to force observations in forming the likelihood. If None, force observations will be ommitted
        
        stress_weight: float or None, optional
            Weight given to stress observations in forming the likelihood. If None, stress observations will be ommitted
        
        sqrt_prior : np.array of float, optional
            Square root of the prior matrix, used to form the posterior. If None, an identity matrix is used.

        lowmem : bool, optional
            Whether to use the low memory variant of solving for the posterior. Cannot be changed after initialisation.
            Defaults to False
        
        random_seed : int or np.random.RandomState object
            Seed or random state to use for all random processes.

        regularisation: float, optional
            Regularisation strength used to ensure likelihood is positive definite in the low memory variant.
            Used in a cholesky decomposition cholesky(likelihood + regularisation * np.eye(self.n_desc)) to obtain
            a square root of the likelihood. Default is 1e-4

        '''

        self.comm = mpi_comm

        if self.comm is not None:
            if not has_mpi or type(self.comm) not in  [MPI.Comm, MPI.Intracomm]:
                raise RuntimeError("mpi_comm argument passed without a valid MPI4Py Communicator")
            
            self.rank = self.comm.Get_rank()
            self.comm_size = self.comm.Get_size()
        else:
            self.rank = None
            self.comm_size = None

        super().__init__(**kwargs)

        self.regularisation = regularisation

        self.n_comm = committee_size
        self.n_desc = descriptor_size

        self.selected_structures = []

        self.weights = [None, None, None]
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight

        self.prior_weight = prior_weight

        if sqrt_prior is None:
            self.sqrt_prior = np.eye(self.n_desc, self.n_desc)
        else:
            self.sqrt_prior = sqrt_prior
        
        self._lowmem = lowmem

        if self._lowmem:
            self.likelihood = {key : np.zeros_like(self.sqrt_prior) for key in ["energy", "force", "stress"]}
        else:
            self.likelihood = {key : [] for key in ["energy", "force", "stress"]}

        if random_seed is not None:
            if type(random_seed) == np.random.RandomState:
                self.rng = random_seed
            elif type(random_seed) == int:
                self.rng = np.random.RandomState(seed=random_seed)
            else:
                raise RuntimeError("random_seed must be an integer, or a np.random.RandomState object")
        else:
            self.rng = np.random.RandomState()

        self.committee_weights = None

    @property
    def energy_weight(self):
        return self.weights[0]
    
    @property
    def forces_weight(self):
        return self.weights[1]
    
    @property
    def stress_weight(self):
        return self.weights[2]
    
    @energy_weight.setter
    def energy_weight(self, weight):
        if weight is not None:
            assert weight > 0

        self.weights[0] = weight

    @forces_weight.setter
    def forces_weight(self, weight):
        if weight is not None:
            assert weight > 0

        self.weights[1] = weight

    @stress_weight.setter
    def stress_weight(self, weight):
        if weight is not None:
            assert weight > 0
    
        self.weights[2] = weight

    @abstractmethod
    def get_descriptor_energy(self, atoms):
        '''
        Get "descriptor energy", which is the average of descriptor vectors in the structure

        Returns an array of length self.n_desc
        '''
        pass

    @abstractmethod
    def get_descriptor_force(self, atoms):
        '''
        Get "descriptor forces", which are the derivatives w.r.t atomic positions of the total descriptor energy

        Returns an array of shape (self.n_desc, Nats, 3)
        '''
        pass

    @abstractmethod
    def get_descriptor_stress(self, atoms):
        '''
        Get "descriptor stresses", which are the stress analogues to the total descriptor energy

        Returns an array of shape (self.n_desc, 9)
        '''
        pass

    def get_committee_energies(self, atoms=None):
        '''
        Get energy predictions by each member of the committee

        Returns an array of length self.n_comm
        
        '''
        if atoms is None:
            atoms = self.atoms

        d = self.get_descriptor_energy(atoms)

        return self.committee_weights @ d
    
    def get_committee_forces(self, atoms=None):
        '''
        Get force predictions by each member of the committee

        Returns an array of shape (self.n_comm, Nats, 3)
        
        '''
        
        if atoms is None:
            atoms = self.atoms
        
        d = self.get_descriptor_force(atoms)

        return self.committee_weights @ d
    
    def get_committee_stresses(self, atoms=None):
        '''
        Get stress predictions by each member of the committee

        Returns an array of shape (self.n_comm, 9)

        
        '''

        if atoms is None:
            atoms = self.atoms

        d = self.get_descriptor_stress(atoms)

        return self.committee_weights @ d
    
    def calculate(self, atoms, properties, system_changes):
        '''
        Normal calculation, using committee averaged properties
        
        '''
        super().calculate(atoms, properties, system_changes)

        if "energy" in properties:
            self.results["energy"] = np.mean(self.get_committee_energies(atoms))
        
        if "forces" in properties:
            self.results["forces"] = np.mean(self.get_committee_forces(atoms), axis=0)

        if "stress" in properties:
            self.results["stress"] = np.mean(self.get_committee_stresses(atoms), axis=0)

    def hal_calculate(self, atoms, properties, system_changes):
        '''
        Calculate the energy, force, and stress properties from the committee used for HAL

        Energy is the std() of the committee energies
        Forces and stresses are weighted averages, using weights of the energy predictions
        Stresses
        
        '''
        super().calculate(atoms, properties, system_changes)

        if "energy" in properties or "forces" in properties:
            Es = self.get_committee_energies(atoms)

        if "energy" in properties:
            self.results["hal_energy"] = np.std(Es)

        if "forces" in properties:
            Es -= np.mean(Es)
            Fs = self.get_committee_forces(atoms)
            Fs -= np.mean(Fs, axis=0)

            self.results["hal_forces"] = np.mean([E * F for E, F in zip(Es, Fs)], axis=0)

        if "stress" in properties:
            Ss = self.get_committee_stresses(atoms)
            Ss -= np.mean(Ss, axis=0)

            self.results["hal_stresses"] = np.mean([E * S for E, S in zip(Es, Ss)], axis=0)

    def __update_likelihood_core(self, atoms, energy_weight, force_weight, stress_weight):
        l = {}

        if energy_weight is not None:
            l["energy"] = np.sqrt(energy_weight) * self.get_descriptor_energy(atoms)[None, ...]

        if force_weight is not None:
            l["force"] = np.sqrt(force_weight) * self.get_descriptor_force(atoms).reshape(self.n_desc, -1).T

        if stress_weight is not None:
            l["stress"] = np.sqrt(stress_weight) * self.get_descriptor_stress(atoms)


        if self._lowmem:
            # Low memory variant
            # Setup problem as Phi^T Phi + Prior

            for key in ["energy", "force", "stress"]:
                if key in l.keys():
                    self.likelihood[key] += l[key].T @ l[key]
        
        else:
            # Normal variant
            # Assemble list of all observations
            # Maintain as list to not shift results around in memory until needed.
            for key in ["energy", "force", "stress"]:
                if key in l.keys():
                    self.likelihood[key].extend(l[key])


    def _update_likelihood(self, atoms):
        '''
        Update the likelihood based on observations obtained from atoms

        Parameters
        ----------
        atoms: ase.atoms.Atoms object
            Atoms object to derive energy, force, and stress observations from
        
        '''
        self.__update_likelihood_core(atoms, *self.weights)

    def _MPI_broadcast_selection(self, atoms):
        '''
        Use MPI4Py to send a selected atoms object to all other processes, along with a 
        snapshot of the current energy, force, and stress weights
        
        Skipped if not enabled
        '''

        data = [atoms.copy(), *self.weights]

        if has_mpi and self.comm is not None:
            for i in range(self.comm_size):
                if i != self.rank:
                   self.comm.isend(data, dest=i) # Non-blocking broadcast to each other process
    
    def _MPI_receive_all_selections(self):
        '''
        Use MPI4Py to receive all structures selected by other processes, along with the energy, force
        and stress weights at the time the structures were selected

        Skipped if MPI not enabled
        '''
        if not (has_mpi and self.comm is not None):
            return # MPI not enabled, so skip this step
        
        req = self.comm.irecv()

        while req.get_status(): # True when message is pending
            st, msg = req.test() # Recieve status and message via request


            self.selected_structures.append(msg[0].copy()) # Add selected structure to list
            self.__update_likelihood_core(*msg)

            req.free() # Close and open a new request, for the new message
            req = self.comm.irecv()

        req.free()
    
    def sync(self):
         if not (has_mpi and self.comm is not None):
            return # MPI not enabled, so skip this step
         
         self.comm.barrier() # MPI barrier to make sure all ranks are together
         self._MPI_receive_all_selections()
         self.comm.barrier() # 2nd barrier to ensure all messages are recieved
    
    def select_structure(self, atoms):
        self.selected_structures.append(atoms.copy())
        self._update_likelihood(atoms)

        self._MPI_broadcast_selection(atoms) # Send selected structure to all other processes, to be picked up later

    def resample_committee(self, committee_size=None):
        '''
        Resample the committee, based on the states of self.likelihood and self.sqrt_prior
        Populates self.committee_weights based on the newly sampled committee

        Parameters
        ----------
        committee_size : int, optional
            New size of the committee, if supplied.
            By default, a committee of size self.n_comm is drawn

        '''
        self._MPI_receive_all_selections() # Sync up with selections from other processes


        if committee_size is not None:
            self.n_comm = committee_size


        if self._lowmem:
            L_likelihood = np.linalg.cholesky(np.sum([self.likelihood[key] for key in ["energy", "force", "stress"]]) 
                                              + self.regularisation * np.eye(self.n_desc))

            sqrt_posterior = L_likelihood + np.sqrt(self.prior_weight) * self.sqrt_prior

            Q, R = np.linalg.qr(sqrt_posterior)
        
        else:
            l_list = []

            for key in ["energy", "force", "stress"]:
                l_key = self.likelihood[key]
                if len(l_key):
                    l_list.extend(l_key)
            sqrt_posterior = np.vstack(l_list + [self.sqrt_prior])
            Q, R = np.linalg.qr(sqrt_posterior)

        
        z = self.rng.normal(loc=0, scale=1, size=(self.n_desc, self.n_comm))

        self.committee_weights = solve_triangular(R, z, lower=False).T # zero mean committee, so no mean term

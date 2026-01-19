Developer Guide
===============

This page details how to contribute to ase_uhal. 



Implementing New MLIP Descriptors
---------------------------------
Implementation of new MLIP Descriptors is made simple through the base classes ``BaseCommitteeCalculator`` and ``TorchCommitteeCalculator``.
``TorchCommitteeCalculator`` itself inherits from ``BaseCommitteeCalculator``, and features torch-compatible solves of the linear system, as well as an extension
to self.get_property() which allows the self.results dict to contain torch.Tensors rather than np.arrays (so that pre-calculated results may be later autodiffed)

The basic requirement for all subclasses of ``BaseCommitteeCalculator`` is to implement the ``self.calculate()`` (as with any ASE calculator). However,
we extend the list of implemented properties to allow descriptor vectors and biasing properties to also use the same interface. If a subclassing calculator implements
energy calculation, it should also at minimum also implement energy descriptors through the ``desc_energy`` property, and biasing energy through the ``bias_energy`` property. 
The same is true for forces and stresses, if they are also implemented by the calculator. ``ACEHALCalculator`` and ``MACEHALCalculator`` also implement committee properties through
the keys ``comm_energy``, ``comm_forces``, and ``comm_stress``, as these are useful to have, but these properties are not essential for ``BiasCalculator`` to function.


It is strongly recommended that the bias properties are implemented similarly to ``ACEHALCalculator`` / ``BaseACECommitteeCalculator`` and ``MACEHALCalculator`` \ ``BaseMACECommitteeCalculator``,
where an abstract base class (e.g. ``BaseACECommitteeCalculator``) implements the ``self.calculate()`` method, but makes calls to ``abstractmethod``s to calculate the bias properties - 
``BaseACECommitteeCalculator`` requires the ``self._bias_energy()`, ``self._bias_forces()`` and ``self._bias_stress()`` methods to be implemented, but ``BaseMACECommitteeCalculator`` only requires ``self._bias_energy()`` 
as the bias forces and stresses are achieved through torch autodiff. This allows specific biasing potentials (e.g. ``HALBiasPotential`` and ``TorchHALBiasPotential``) to be applied on top of the base class, rather than a reimplementation being required.

.. code-block:: python
    :caption: Sketch of recommended minimal implementation (energy only, but could be extended to forces)

    class NewBaseDescriptorCommitteeCalculator(BaseCommitteeCalculator):
        implemented_properties = ...
        def __init(self, *args, **kwargs):
            ... # Calculate descriptor vector length here, and also call super().__init__()

        def calculate(self, atoms, properties, system_changes):
            super().calculate(atoms, properties, system_changes)

            if "desc_energy" in properties:
                self.results["desc_energy"] = ... # Calculate energy descriptor vector here

            if "bias_energy" in properties:
                comm_energies = ... # Energy prediction for each committee member
                self.results["bias_energy"] = self._bias_energy(comm_energies)

        @abstractmethod
        def _bias_energy(self, comm_energy):
            pass

    class NewDescriptorHALCalculator(NewBaseDescriptorCalculator, HALBiasPotential):
        # HALBiasPotential implements self_bias_energy()
        pass

Subclasses should also calculate the length of the descriptor vector in their ``__init__`` method, and pass this on to ``BaseCommitteeCalculator``. For subclasses of ``TorchCommitteeCalculator``,
this should be achieved through the implementation of a ``self._get_descriptor_length()`` function, which ``TorchCommitteeCalculator`` calls during ``TorchCommitteeCalculator.__init__()`` -
this is so that the descriptor length calculation happens after torch is imported by ``TorchCommiteeCalculator``, but before ``BaseCommitteeCalculator.__init__()`` is called.


Implementing New Bias Potentials
--------------------------------
Biasing potentials other than the HAL approach can be implemented using a class which subclasses either ``BaseACECommitteeCalculator`` or ``BaseMACECommitteeCalculator``, with the specific implementation of the biasing potential.
Subclasses of ``BaseACECommitteeCalculator`` should implement the ``self._bias_energy()`, ``self._bias_forces()`` and ``self._bias_stress()`` methods, whilst subclasses of ``BaseMACECommitteeCalculator`` only need to implement ``self._bias_energy()`` (as 
the bias forces and stresses can be found using torch.autodiff).

For interoperability with any new MLIP descriptor classes, the HAL bias potential was implemented as a separate base class. This allows the specific ACE+HAL calculator to be specified via double inheritance:

.. code-block:: python

    class ACEHALCalculator(BaseACECalculator, HALBiasPotential):
       name = "ACEHALCalculator"

A similar design pattern is recommended for new biasing potentials as well. Both CPU-only and torch-compatible versions of the bias potential should also be specified, so that torch remains an optional 
dependency only for the MLIP models which already rely on it.
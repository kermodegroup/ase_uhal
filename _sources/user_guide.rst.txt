User Guide
===============

Some prerequisite :ref:`theory` is assumed to fully understand this User Guide. The guide provides an overview of what the code
provides, but it is strongly recommended to follow the tutorials to understand how it works in practice.

Committee Calculators
---------------------
The committee calculators provide an interface to a given models descriptor (e.g. the ACE descriptor, or the ``mace.calculators.MACECalculator.get_descriptor()`` method)
and allow for the construction of committees of linear models based on this descriptor.


.. warning::

    It is not recommended to use the committee calculators as "normal" ASE calculators, although they do function in that role. 
    The "energy", "forces", and "stress" proprerties are implemented as the committee mean properties (so that the calculator functions more like
    a conventional committee model), which should all be close to zero due to the use of zero-mean committees.

    To access the biasing variant of each property, we define new "bias_energy", "bias_forces", and "bias_stress" properties. These can be easily obtained using
    the ``get_property`` function, e.g.:

    .. code:: python
        
        comm_calc = ACEHALCalculator(...)
        E_bias = comm_calc.get_property("bias_energy", atoms)


ase_uhal.committee_calculators.ACEHALCalculator
++++++++++++++++++++++++++++++++++++++++++++++++




ase_uhal.committee_calculators.MACEHALCalculator
++++++++++++++++++++++++++++++++++++++++++++++++


Bias Calculators
-----------------
The bias calculators implement biasing between a mean calculator (e.g an existing ACE model, or a MACE foundation model) and a committee calculator.


ase_uhal.bias_calculators.HALBiasCalculator
+++++++++++++++++++++++++++++++++++++++++++



ase_uhal.Structure Selector
---------------------------


Running Biased Dynamics
-----------------------



ase_uhal in parallel
--------------------
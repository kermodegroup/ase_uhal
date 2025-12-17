User Guide
===============

Some prerequisite :ref:`theory` is assumed to fully understand this User Guide. The guide provides an overview of what the code
provides, but it is strongly recommended to follow the tutorials to understand how it works in practice.

Committee Calculators
---------------------
The committee calculators provide an interface to a given models descriptor (e.g. the ACE descriptor, or the ``mace.calculators.MACECalculator.get_descriptor()`` method)
and allow for the construction of committees of linear models based on this descriptor.

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
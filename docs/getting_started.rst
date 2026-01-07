Getting Started
===============

``ase_uhal`` Overview
----------------------
The code provides two kinds of ASE-compatible calculators:

- The committee calculators provide interfaces for committee models using descriptors from various MLIP models (currently supported: ACE and MACE). 
- The bias calculators implement energy, force, and stress biasing which combines the "true" properties from a mean_calculator (e.g. from a MACE foundation model) with the "bias" properties as calculated by the committee calculator.

The bias calculator can then be used like any other ASE calculator to run (biased) Molecular Dynamics. 
Using ``ase_uhal.ACEHALCalculator`` with ``ase_uhal.HALBiasCalculator`` provides a close approximation to the original `ACEHAL <https://github.com/ACEsuit/ACEHAL/tree/main>`__ approach.

See the User Guide for more details on this.

Installation
------------

Installation of the main module can be achieved using pip:

.. code-block:: bash
    
    pip install ase_uhal

MACE
-----
For MACE-based biasing, the ``mace-torch`` Python package must be also installed. As a shortcut, this can be done using the ``[mace]`` optional dependancies of this package, e.g.

.. code-block:: bash

    pip install ase_uhal[mace]

ACE
----
For ACE-based biasing, the ``juliacall`` package must be installed, along with the ACEpotentials, AtomsBase, and Unitful Julia dependencies.
This can easily be achieved through the following:

.. code-block:: bash

    pip install ase_uhal[ace]
    python -c "import ase_uhal; ase_uhal.install_ace_deps()"

Customising the Julia Environment
++++++++++++++++++++++++++++++++++
The ACE/Julia compatibility is provided using the `juliapkg <https://github.com/JuliaPy/PyJuliaPkg>`__ Python module. This will generally attempt to install the required packages 
as purely as possible (i.e. trying not to generate side effects in existing julia installs and projects), including installing the Julia Project.toml inside a Python
virtual enviroment, if one exists and is activated when ``ase_uhal.install_ace_deps()`` is called.

By default, ``juliapkg`` will look for the ``julia`` exe in ``PATH``, and will use a blank Julia project regardless of whether ``JULIAPROJECT`` is set.
We can however customise this using environment variables (taken from the ``juliapkg`` docs):

.. code-block:: bash

    # Path to Julia exe file
    export PYTHON_JULIAPKG_EXE=.

    # Path to Project.toml to use/create when resolving deps
    export PYTHON_JULIAPKG_PROJECT=.

.. warning::
    Setting ``PYTHON_JULIAPKG_PROJECT`` to an existing Project.toml and then running ``ase_uhal.install_julia_deps()`` may modify the existing
    environment, if it was not already compatible.
Getting Started
===============

Installation
------------

Installation of the main module can be achieved using pip from the top-level ase-uhal directory:

.. code-block:: bash
    
    pip install .

For MACE-based biasing, the `mace-torch` Python package must be also installed. As a shortcut, this can be done using the `[mace]` optional dependancies of this package, e.g.

.. code-block:: bash

    pip install .[mace]

For ACE-based biasing, the julia package must be installed, along with the ACEpotentials, AtomsBase, and Unitful Julia dependencies.

.. code-block:: bash

    pip install .[ace] # Installs the julia package
    python -c "import julia; julia.install()" # Sets up link between Python and Julia
    julia # Opens Julia repl

Then inside the Julia repl:

.. code-block:: julia
    
    using Pkg
    Pkg.add("ACEpotentials")
    Pkg.add("AtomsBase")
    Pkg.add("Unitful")

ase_uhal Overview
-----------------
The code provides two kinds of ASE-compatible calculators:

- The committee calculators provide interfaces for committee models using descriptors from various MLIP models (currently supported: ACE and MACE). 
- The HAL calculator implements energy, force, and stress biasing which combines the "true" properties from a mean_calculator (e.g. from a MACE foundation model) with the "HAL" properties as calculated by the committee calculator.

The HAL calculator can then be used like any other ASE calculator to run (biased) Molecular Dynamics.

See the User Guide for more details on this.
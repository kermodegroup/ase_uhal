ase_uhal
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/thomas-rocke/ase_uhal/workflows/CI/badge.svg)](https://github.com/thomas-rocke/ase_uhal/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/thomas-rocke/ase_uhal/branch/main/graph/badge.svg)](https://codecov.io/gh/thomas-rocke/ase_uhal/branch/main)


Implementation of "Universal HyperActive Learning" compatible with the Atomic Simulation Environment (ASE)

### Documentation
Documentation is available at https://kermodegroup.github.io/ase_uhal/


### Installation
Requires:
- Python >= 3.10
- Julia >= 1.11 (for ACE descriptor features)

Basic installation can be achieved by Git cloning this repository, and installing via pip:
```bash
git clone https://github.com/kermodegroup/ase_uhal.git
cd ase_uhal
pip install .
```
Interfaces to MLIP descriptors are handled as optional dependencies to this package. For example, to install the MACE compatibility,
```bash
pip install .[mace]
```

#### ACE Installation
ACE installation is more complex, as it requires a connection between Python and Julia, both with the correct modules installed. This is handled by `pyjuliapkg`, and can be achieved via:
```bash
pip install .[ace]
python -c "import ase_uhal; ase_uhal.install_ace_deps()"
```
For more details on this, including customising the Julia installation, see the documentation.

### Copyright

Copyright (c) 2025, Thomas Rocke


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.

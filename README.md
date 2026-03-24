ase_uhal
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/kermodegroup/ase_uhal/workflows/CI/badge.svg)](https://github.com/kermodegroup/ase_uhal/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/kermodegroup/ase_uhal/branch/main/graph/badge.svg)](https://codecov.io/gh/kermodegroup/ase_uhal/branch/main)


Implementation of "Universal HyperActive Learning" compatible with the Atomic Simulation Environment (ASE)

### Documentation
Documentation is available at https://kermodegroup.github.io/ase_uhal/


### Base Installation
Requires:
- Python >= 3.10
- Julia >= 1.11 (for ACE descriptor features)

**_NOTE:_**  This package is intended to always be used in conjunction with additional dependencies, see the next section for more information.

Basic installation can be achieved via pip
```bash
pip install ase-uhal

```

You can also clone the Git repository:
```bash
git clone https://github.com/kermodegroup/ase_uhal.git
cd ase_uhal
pip install .
```

### Extra Dependencies
`ase_uhal` supports interfaces to multiple MLIP architectures. To avoid a very large number of mandatory dependencies, specific requirements for each MLIP model are implemented as optional dependencies to this package. For example, to install the MACE compatibility,
```bash
pip install ase-uhal[mace]
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

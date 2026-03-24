# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ase_uhal** implements Universal HyperActive Learning (UHAL) for the Atomic Simulation Environment (ASE). It provides committee-based uncertainty quantification for machine learning interatomic potentials (MLIPs), enabling active learning during molecular dynamics simulations. Supports MACE (PyTorch) and ACE (Julia) backends.

## Build & Install

```bash
uv sync --extra test                # Install with all test dependencies (includes mace + ace)
uv run python -c "import ase_uhal; ase_uhal.install_ace_deps()"  # Install Julia/ACE deps (one-time, requires Julia >= 1.11)
```

Julia packages are installed into `.venv/julia_env/` by `install_ace_deps()`. This step is required for ACE tests to pass — without it, only MACE tests will work.

## Testing

```bash
uv run pytest -v                                       # Run all tests
uv run pytest -v ase_uhal/tests/test_committee_calc.py # Committee calculator tests
uv run pytest -v ase_uhal/tests/test_bias_calcs.py     # Bias calculator tests
uv run pytest -v ase_uhal/tests/test_committee_calc.py::TestCommitteeCalcs::test_force_derivative  # Single test class method
uv run pytest -v ase_uhal/tests/test_committee_calc.py -k "ACE1Calculator"  # Filter by keyword
uv run pytest -v --cov=ase_uhal --cov-report=xml ase_uhal/tests/  # With coverage (CI mode)
```

Tests are parametrized across calculator backends (MACE, ACE) and validate correctness via finite difference checks against analytical forces/stress.

## Code Style

- **Formatter:** YAPF with 119 column limit, 4-space indent
- **Linter:** Flake8 with 119 character line limit
- Configuration in `setup.cfg`

## Architecture

### Calculator Hierarchy

The core abstraction is ASE-compatible calculators organized in an inheritance hierarchy:

```
BaseCommitteeCalculator (ase_uhal/committee_calculators/base_committee_calculator.py)
├── Bayesian linear regression posterior over descriptor space
├── Committee weight sampling from posterior Gaussian (QR decomposition)
├── MPI support for distributed computation
│
├── TorchCommitteeCalculator (torch_committee_calculator.py)
│   └── BaseMACECalculator → MACEHALCalculator (mace_committee_calculator.py)
│       Extracts descriptors from MACE model layers, GPU support via PyTorch
│
└── BaseACECalculator → ACEHALCalculator (ace_committee_calculator.py)
    Julia interop via juliacall, descriptor evaluation via ACEpotentials
```

### Bias Calculators (`bias_calculators.py`)

`HALBiasCalculator` wraps a mean calculator + committee calculator. It computes biased energy/forces/stress using committee disagreement (uncertainty), with adaptive `tau` parameter controlling bias strength.

### Structure Selection (`structure_selector.py`)

`StructureSelector` is an MD observer that monitors HAL score (committee disagreement) during dynamics. When uncertainty exceeds an adaptive threshold, it selects the current structure for training data and triggers committee resampling.

### Distillation (`distillation/`)

Assembles a linear system from descriptors and reference data to "distill" committee models. Entry points: `distill_dataset()` and `estimate_memory_spike()`.

### Julia Integration

ACE backend uses `juliacall` to interface with Julia's ACEpotentials. Julia utilities live in `ase_uhal/data/_ace_utils.jl`. Julia dependencies are declared in `Project.toml`.

## Key Patterns

- All calculators implement the ASE `Calculator` interface (`calculate()` method with `properties` and `system_changes` args)
- Committee uncertainty = standard deviation across committee member predictions
- `resample_committee()` rebuilds posterior and draws new weight samples after training data changes
- Tests use `pytest_allclose` fixture for numerical tolerance and central finite differences for derivative validation

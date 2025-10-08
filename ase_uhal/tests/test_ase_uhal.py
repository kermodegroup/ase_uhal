"""
Unit and regression test for the ase_uhal package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import ase_uhal


def test_ase_uhal_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "ase_uhal" in sys.modules

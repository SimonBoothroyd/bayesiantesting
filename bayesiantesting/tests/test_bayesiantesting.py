"""
Unit and regression test for the bayesiantesting package.
"""

# Import package, test suite, and other packages as needed
import bayesiantesting
import pytest
import sys


def test_bayesiantesting_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "bayesiantesting" in sys.modules

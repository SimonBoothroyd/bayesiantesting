"""
Unit and regression test for the datasets module.
"""

import pytest

from bayesiantesting.datasets.nist import NISTDataSet


@pytest.mark.parametrize(
    "compound",
    ["Br2", "C2Cl4", "C2F4", "C2H2", "C2H4", "C2H6", "Cl2", "F2", "N2", "O2"],
)
def test_nist_data_sets(compound):
    NISTDataSet(compound)

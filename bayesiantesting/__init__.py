"""
bayesiantesting
A private space for testing out projects based on Bayesian methods
"""

# Set up pint.
from pint import UnitRegistry

unit = UnitRegistry()

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

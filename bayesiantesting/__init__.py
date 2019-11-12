"""
bayesiantesting
A private space for testing out projects based on Bayesian methods
"""

from pint import UnitRegistry

from ._version import get_versions

# Set up pint.
unit = UnitRegistry()

# Handle versioneer
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

"""
bayesiantesting
A private space for testing out projects based on Bayesian methods
"""

# Add imports here
from .bayesiantesting import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

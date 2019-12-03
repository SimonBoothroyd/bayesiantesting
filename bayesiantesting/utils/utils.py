"""
This code is based upon the implementations by Owen Madin
"""
import contextlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymbar
import scipy as sp
import yaml
from scipy.optimize import curve_fit
from scipy.stats import expon
from statsmodels.stats.proportion import multinomial_proportions_confint


def get_data_filename(relative_path):
    """Get the full path to one of the reference files in data.

    In the source distribution, these files are in ``bayesiantesting/data/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    relative_path : str
        The relative path of the file to load.
    """

    from pkg_resources import resource_filename

    fn = resource_filename("bayesiantesting", os.path.join("data", relative_path))

    if not os.path.exists(fn):
        raise ValueError(
            "Sorry! %s does not exist. If you just added it, you'll have to re-install"
            % fn
        )

    return fn


@contextlib.contextmanager
def temporarily_change_directory(file_path):
    """A context to temporarily change the working directory.

    Parameters
    ----------
    file_path: str
        The file path to temporarily change into.
    """
    prev_dir = os.getcwd()
    os.chdir(os.path.abspath(file_path))

    try:
        yield
    finally:
        os.chdir(prev_dir)

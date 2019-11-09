import os


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
    fn = resource_filename('bayesiantesting', os.path.join('data', relative_path))

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn

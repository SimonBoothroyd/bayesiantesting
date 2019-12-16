#!/usr/bin/env python3
import functools
import json
import os
from multiprocessing.pool import Pool

import numpy

from bayesiantesting.kernels.rjmc import WidomRJMC
from bayesiantesting.models.continuous import UnconditionedModel
from bayesiantesting.models.discrete import TwoCenterLJModelCollection
from bayesiantesting.utils import distributions
from studies.utilities import (
    fit_to_trace,
    get_2clj_model,
    parse_input_yaml,
    prepare_data,
)


def fit_distributions(models, simulation_params, n_processes):
    """Fit 1D analytical distributions to the posteriors of
    a set of models from short MCMC simulations.

    Parameters
    ----------
    models: list of Model
        The models to fit the posteriors of.
    simulation_params: dict
        The simulation parameters.
    n_processes: int
        The number of process to run on.

    Returns
    -------
    list of list of Distribution
        The fit distributions for each model.
    list of numpy.ndarray
        The MAP of each model as determined from the
        fitted distributions.
    """
    fitting_directory = os.path.join(simulation_params["compound"])
    os.makedirs(fitting_directory, exist_ok=True)

    initial_parameters = {
        "UA": numpy.array([100.0, 0.4]),
        "AUA": numpy.array([100.0, 0.4, 0.15]),
        "AUA+Q": numpy.array([136.0, 0.321, 0.21, 0.01]),
    }

    with Pool(n_processes) as pool:

        pool.map(
            functools.partial(
                fit_to_trace,
                warm_up_steps=int(simulation_params["fit_steps"] * 2.0 / 3.0),
                steps=simulation_params["fit_steps"],
                output_directory=fitting_directory,
                initial_parameters=initial_parameters,
            ),
            models,
        )

    # Load the mapping distributions
    mapping_distributions = []
    maximum_a_posteriori = []

    for model in models:

        fit_path = os.path.join(fitting_directory, f"{model.name}_univariate_fit.json")

        with open(fit_path) as file:
            fit_priors = json.load(file)

        fit_model = UnconditionedModel(model.name, fit_priors, {})
        mapping_distributions.append(fit_model.priors)

        # Determine the maximum a posteriori of the fit
        map_parameters = []

        for distribution in fit_model.priors:

            if isinstance(distribution, distributions.Normal):
                map_parameters.append(distribution.loc)
            else:
                raise NotImplementedError()

        maximum_a_posteriori.append(numpy.asarray(map_parameters))

    return mapping_distributions, maximum_a_posteriori


def main(n_processes=1):

    # Load in the simulation parameters.
    simulation_params = parse_input_yaml("widom_run.yaml")

    # Load the data.
    data_set, property_types = prepare_data(simulation_params)

    # Build the model / models.
    sub_models = [
        get_2clj_model("AUA", data_set, property_types, simulation_params),
        get_2clj_model("AUA+Q", data_set, property_types, simulation_params),
        get_2clj_model("UA", data_set, property_types, simulation_params),
    ]

    # Fit an analytical distribution to the traces of short MCMC
    # simulation of each model. These will be used to construct the
    # cross-model proposals.
    mapping_distributions, maximum_a_posteriori = fit_distributions(
        sub_models, simulation_params, n_processes
    )

    # Create the full model collection
    model_collection = TwoCenterLJModelCollection(
        "2CLJ Models", sub_models, mapping_distributions
    )

    # Define the model index to make the 'proposals' from.
    initial_model_index = 1
    initial_model = model_collection.models[initial_model_index]
    initial_parameters = maximum_a_posteriori[initial_model_index]

    output_directory_path = simulation_params["compound"] + f"_{initial_model.name}"

    simulation = WidomRJMC(
        model_collection=model_collection,
        initial_parameters=initial_parameters,
        initial_model_index=initial_model_index,
        swap_frequency=simulation_params["swap_freq"],
    )

    simulation.run(
        warm_up_steps=int(simulation_params["steps"] * 0.1),
        steps=simulation_params["steps"],
        output_directory=output_directory_path,
    )

    # TODO: Apply EXP / BAR.
    # proposal_alphas = simulation.proposal_trace


if __name__ == "__main__":
    main()

#!/usr/bin/env python3# from bayesiantesting.kernels import MCMCSimulation
import argparse
import functools
import json
import os
from multiprocessing.pool import Pool

import numpy

from bayesiantesting.kernels.bayes import MBARIntegration
<<<<<<< HEAD
from studies.utilities import (
    fit_to_trace,
    get_2clj_model,
    parse_input_yaml,
    prepare_data,
)
=======
from bayesiantesting.models.continuous import UnconditionedModel
from studies.utilities import get_2clj_model, parse_input_yaml, prepare_data


def prior_dictionary_to_json(prior_settings, file_path):
    """Saves a dictionary of prior settings to JSON.

    Parameters
    ----------
    prior_settings: dict
        The dictionary of settings.
    file_path: str
        The file path to save the JSON to.
    """

    json_dictionary = {}

    for label in prior_settings:

        json_label = label

        if isinstance(json_label, tuple):
            json_label = ",".join(json_label)

        json_dictionary[json_label] = [prior_settings[label][0], []]

        for parameter in prior_settings[label][1]:

            parameter_list = parameter.tolist()
            json_dictionary[json_label][1].append(parameter_list)

    with open(file_path, "w") as file:
        json.dump(
            json_dictionary, file, sort_keys=False, indent=4, separators=(",", ": ")
        )


def prior_dictionary_from_json(file_path):
    """Loads a dictionary of prior settings from a JSON
    file.

    Parameters
    ----------
    file_path: str
        The file path to load the JSON from.

    Returns
    -------
    dict
        The dictionary of prior settings.
    """

    with open(file_path, "r") as file:
        json_dictionary = json.load(file)

    prior_settings = {}

    for json_label in json_dictionary:

        label = json_label

        if label.find(",") >= 0:
            label = tuple(label.split(","))

        prior_settings[label] = [json_dictionary[json_label][0], []]

        for parameter_list in json_dictionary[json_label][1]:

            parameter = parameter_list

            if isinstance(parameter, list):
                parameter = numpy.asarray(parameter_list)

            prior_settings[label][1].append(parameter)

    return prior_settings


def fit_prior_to_trace(parameter_trace):
    """Fits either a normal or a half normal distribution
    to a given trace.

    Parameters
    ----------
    parameter_trace: numpy.ndarray
        The parameter trace from an `MCMCSimulation` simulation
        with shape=(n_steps).

    Returns
    -------
    list
        The prior settings.
    """
    loc = numpy.mean(parameter_trace)
    scale = numpy.std(parameter_trace)

    if loc - 5.0 * scale > 0.0:
        # Check to make sure whether a half-normal distribution may be
        # more appropriate.
        return ["normal", [loc, scale]]

    scale = numpy.sqrt(numpy.sum(parameter_trace ** 2) / len(parameter_trace))
    return ["half normal", [scale]]


def fit_to_trace(model, output_directory, initial_parameters, use_existing=True):
    """Fits a multivariate gaussian distribution to the posterior
    of the model as sampled by an MCMC simulation.

    Parameters
    ----------
    model: Model
        The model to sample.
    output_directory: str
        The directory to store the working files in.
    initial_parameters: numpy.ndarray
        The parameters to start the simulation from.
    use_existing: bool
        If True, any existing fits will be used rather than regenerating
        new fits.

    Returns
    -------
    UnconditionedModel
        The fitted univariate model.
    UnconditionedModel
        The fitted multivariate model.
    """

    trace_path = os.path.join(output_directory, model.name, "trace.npy")

    if not use_existing or not os.path.isfile(trace_path):

        # initial_parameters = generate_initial_parameters(model)
        initial_parameters = initial_parameters[model.name]

        # Run a short MCMC simulation to get better initial parameters
        simulation = MCMCSimulation(
            model_collection=model, initial_parameters=initial_parameters,
        )

        simulation.run(
            warm_up_steps=10000, steps=15000, output_directory=output_directory
        )

    trace = numpy.load(trace_path)

    # Fit the univariate distributions.
    univariate_prior_dictionary = {}

    for index, label in enumerate(model.trainable_parameter_labels):

        parameter_trace = trace[:, index + 1]
        univariate_prior_dictionary[label] = fit_prior_to_trace(parameter_trace)

    prior_dictionary_to_json(
        univariate_prior_dictionary,
        os.path.join(output_directory, f"{model.name}_univariate_fit.json"),
    )

    # Fit the multivariate distribution.
    n_multivariate_parameters = model.n_trainable_parameters

    if model.name == "AUA+Q":
        n_multivariate_parameters -= 1

    multivariate_mean = numpy.mean(trace[:, 1 : 1 + n_multivariate_parameters], axis=0)
    multivariate_covariance = numpy.cov(trace[:, 1 : 1 + n_multivariate_parameters].T)

    multivariate_key = tuple(
        [
            label
            for label in model.trainable_parameter_labels[:n_multivariate_parameters]
        ]
    )

    multivariate_prior_dictionary = {
        multivariate_key: [
            "multivariate normal",
            [multivariate_mean, multivariate_covariance],
        ]
    }

    for index, label in enumerate(
        model.trainable_parameter_labels[n_multivariate_parameters:]
    ):

        parameter_trace = trace[:, n_multivariate_parameters + index + 1]
        multivariate_prior_dictionary[label] = fit_prior_to_trace(parameter_trace)

    prior_dictionary_to_json(
        multivariate_prior_dictionary,
        os.path.join(output_directory, f"{model.name}_multivariate_fit.json"),
    )

    return (
        UnconditionedModel(f"{model.name}_univariate", univariate_prior_dictionary, {}),
        UnconditionedModel(
            f"{model.name}_multivariate", multivariate_prior_dictionary, {}
        ),
    )
>>>>>>> 225b629657972285766c12a59f15f13faddf462a


def main(compound, n_processes):

    simulation_params = parse_input_yaml("basic_run.yaml")

    # Load the data.
    data_set, property_types = prepare_data(simulation_params, compound)

    # Build the models.
    models = [
        get_2clj_model(model_name, data_set, property_types, simulation_params)
        for model_name in ["UA", "AUA", "AUA+Q"]
    ]

    # Fit multivariate normal distributions to this models posterior.
    # This will be used as the analytical reference distribution.
    fitting_directory = os.path.join(compound, "fitting")
    os.makedirs(fitting_directory, exist_ok=True)

    initial_parameters = {
        "UA": numpy.array([100.0, 0.4]),
        "AUA": numpy.array([100.0, 0.4, 0.15]),
        "AUA+Q": numpy.array([136.0, 0.321, 0.21, 0.01]),
    }

    with Pool(n_processes) as pool:
        all_fits = pool.map(
            functools.partial(
                fit_to_trace,
                output_directory=fitting_directory,
                initial_parameters=initial_parameters,
            ),
            models,
        )

    # Run the MBAR calculations
    results = {}

    for model, fits in zip(models, all_fits):

        _, reference_model = fits

        # Run the MBAR simulation
        lambda_values = numpy.linspace(0.0, 1.0, 3)

        output_directory = os.path.join(results, compound, f"mbar_{model.name}")

        simulation = MBARIntegration(
            lambda_values=lambda_values,
            model=model,
            warm_up_steps=simulation_params["warm_up_steps"],
            steps=simulation_params["steps"],
            output_directory_path=output_directory,
            reference_model=reference_model,
        )

        _, integral, error = simulation.run(
            reference_model.sample_priors(), number_of_processes=n_processes
        )

        results[model.name] = {"integral": integral, "error": error}

    with open(f"mbar_{compound}_results.json", "w") as file:
        json.dump(results, file, sort_keys=True, indent=4, separators=(",", ": "))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Perform an MBAR model evidence calculation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--compound",
        "-c",
        type=str,
        help="The compound to compute the model evidences for.",
        required=True,
    )

    parser.add_argument(
        "--processes",
        "-nproc",
        type=int,
        help="The number processes to run the calculation across.",
        required=False,
        default=1,
    )

    args = parser.parse_args()

    main(args.compound, args.processes)

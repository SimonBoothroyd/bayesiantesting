#!/usr/bin/env python3# from bayesiantesting.kernels import MCMCSimulation
import argparse
import functools
import json
import os
from multiprocessing.pool import Pool

import numpy

from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.kernels.bayes import MBARIntegration
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


def fit_multivariate_to_trace(
    model, output_directory, initial_parameters, use_existing=True
):
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
        The fitted model.
    """

    fit_name = f"{model.name}_fit"
    fit_path = os.path.join(output_directory, f"{fit_name}.json")

    if use_existing and os.path.isfile(fit_path):

        prior_dictionary = prior_dictionary_from_json(fit_path)
        return UnconditionedModel(fit_name, prior_dictionary, {})

    # initial_parameters = generate_initial_parameters(model)
    initial_parameters = initial_parameters[model.name]

    # Run a short MCMC simulation to get better initial parameters
    simulation = MCMCSimulation(
        model_collection=model,
        warm_up_steps=1000000,
        steps=1500000,
        discard_warm_up_data=True,
        output_directory_path=output_directory,
    )

    trace, _, _ = simulation.run(initial_parameters)

    # Fit the multivariate distribution
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

    prior_dictionary = {
        multivariate_key: [
            "multivariate normal",
            [multivariate_mean, multivariate_covariance],
        ]
    }

    for index, label in enumerate(
        model.trainable_parameter_labels[n_multivariate_parameters:]
    ):

        parameter_trace = trace[:, n_multivariate_parameters + index + 1]

        loc = numpy.mean(parameter_trace)
        scale = numpy.std(parameter_trace)
        prior_dictionary[label] = ["normal", [scale]]

        if loc - 5.0 * scale > 0.0:
            # Check to make sure whether a half-normal distribution may be
            # more appropriate.
            continue

        scale = numpy.sqrt(numpy.sum(parameter_trace ** 2) / len(parameter_trace))
        prior_dictionary[label] = ["half normal", [scale]]

    fitted_distribution = UnconditionedModel(fit_name, prior_dictionary, {})

    # Save a copy of the fit
    prior_dictionary_to_json(prior_dictionary, fit_path)

    return fitted_distribution


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
        reference_fits = pool.map(
            functools.partial(
                fit_multivariate_to_trace,
                output_directory=fitting_directory,
                initial_parameters=initial_parameters,
            ),
            models,
        )

    # Run the MBAR calculations
    results = {}

    for model, reference_model in zip(models, reference_fits):

        # Run the MBAR simulation
        lambda_values = numpy.linspace(0.0, 1.0, 8)

        output_directory = os.path.join(compound, f"mbar_{model.name}")

        simulation = MBARIntegration(
            lambda_values=lambda_values,
            model=model,
            warm_up_steps=simulation_params["warm_up_steps"],
            steps=simulation_params["steps"],
            discard_warm_up_data=True,
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

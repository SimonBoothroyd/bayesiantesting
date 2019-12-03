#!/usr/bin/env python3# from bayesiantesting.kernels import MCMCSimulation
import argparse
import functools
import json
import os
from multiprocessing.pool import Pool

import numpy

from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.kernels.bayes import MBARIntegration
from bayesiantesting.models.continuous import MultivariateGaussian
from studies.utilities import get_2clj_model, parse_input_yaml, prepare_data


def fit_multivariate_to_trace(
    model, output_directory, initial_parameters, use_existing=True
):
    """Fits a multivariate gaussian distribution to the posterior
    of the model as sampled by a short MCMC simulation.

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
    MultivariateGaussian
        The fitted multivariate gaussian model.
    """

    fit_path = os.path.join(output_directory, f"{model.name}_fit.json")

    if use_existing and os.path.isfile(fit_path):

        with open(fit_path) as file:
            return MultivariateGaussian.from_json(file.read())

    # initial_parameters = generate_initial_parameters(model)
    initial_parameters = initial_parameters[model.name]

    # Run a short MCMC simulation to get better initial parameters
    simulation = MCMCSimulation(
        model_collection=model,
        warm_up_steps=1000000,
        steps=1000000,
        discard_warm_up_data=True,
        output_directory_path=output_directory,
    )

    trace, _, _ = simulation.run(initial_parameters)

    mean = numpy.mean(trace[:, 1:], axis=0)
    covariance = numpy.cov(trace[:, 1:].T)

    mean_dictionary = {
        label: value for label, value in zip(model.trainable_parameter_labels, mean)
    }

    fitted_distribution = MultivariateGaussian("gaussian", mean_dictionary, covariance)

    with open(fit_path, "w") as file:
        file.write(fitted_distribution.to_json())

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
            reference_model.mean, number_of_processes=n_processes
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

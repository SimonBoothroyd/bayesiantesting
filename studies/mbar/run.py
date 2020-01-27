#!/usr/bin/env python3# from bayesiantesting.kernels import MCMCSimulation
import argparse
import functools
import json
import os
from multiprocessing.pool import Pool

import numpy

from bayesiantesting.kernels.bayes import MBARIntegration
from studies.utilities import (
    fit_to_trace,
    get_2clj_model,
    parse_input_yaml,
    prepare_data,
)


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
        lambda_values = numpy.linspace(0.0, 1.0, 8)

        output_directory = os.path.join(compound, f"mbar_{model.name}")

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

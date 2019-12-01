#!/usr/bin/env python3

import numpy

from bayesiantesting.kernels import MCMCSimulation
from studies.utilities import get_2clj_model, parse_input_yaml, prepare_data


def main():

    print("Parsing simulation params")
    simulation_params = parse_input_yaml("basic_run.yaml")

    # Load the data.
    data_set, property_types = prepare_data(simulation_params)

    # Set some initial parameter close to the MAP taken from
    # other runs of C2H6.
    initial_parameters = {
        "UA": numpy.array([100.0, 0.35]),
        "AUA": numpy.array([120.0, 0.35, 0.12]),
        "AUA+Q": numpy.array([140.0, 0.35, 0.26, 0.05]),
    }

    # Build the model / models.
    for model_name in ["AUA+Q"]:

        model = get_2clj_model(model_name, data_set, property_types, simulation_params)

        # Run the simulation.
        simulation = MCMCSimulation(
            model_collection=model,
            warm_up_steps=int(simulation_params["steps"] * 0.2),
            steps=simulation_params["steps"],
            discard_warm_up_data=True,
        )

        trace, log_p_trace, percent_deviation_trace = simulation.run(
            initial_parameters[model.name]
        )
        model.plot(trace, log_p_trace, percent_deviation_trace, show=True)


if __name__ == "__main__":
    main()

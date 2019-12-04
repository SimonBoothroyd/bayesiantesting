#!/usr/bin/env python3

import numpy

from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.models.continuous import UnconditionedModel
from studies.utilities import parse_input_yaml


def main():

    simulation_params = parse_input_yaml("basic_run.yaml")

    # Set some initial parameter close to the MAP taken from
    # other runs of C2H6.
    initial_parameters = numpy.array([140.0, 0.35, 0.26, 0.01])

    prior_settings = {
        ("epsilon", "sigma", "L"): [
            "multivariate normal",
            [
                numpy.array([122.9351, 0.3581, 0.2068]),
                numpy.array(
                    [
                        [
                            5.514607656954375,
                            -0.003907597860443889,
                            0.011927926543293587,
                        ],
                        [
                            -0.003907597860443889,
                            2.7821697309014395e-06,
                            -8.46201728756609e-06,
                        ],
                        [
                            0.011927926543293587,
                            -8.46201728756609e-06,
                            2.581822857777358e-05,
                        ],
                    ]
                ),
            ],
        ],
        "Q": ["half normal", [0.01]],
    }

    # Build the model / models.
    model = UnconditionedModel("multivariate", prior_settings)

    # Run the simulation.
    simulation = MCMCSimulation(
        model_collection=model,
        warm_up_steps=int(simulation_params["steps"] * 0.2),
        steps=simulation_params["steps"],
        discard_warm_up_data=True,
    )

    trace, log_p_trace, percent_deviation_trace = simulation.run(initial_parameters)
    model.plot(trace, log_p_trace, percent_deviation_trace, show=True)


if __name__ == "__main__":
    main()

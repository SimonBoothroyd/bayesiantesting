#!/usr/bin/env python3

import numpy

from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.models.continuous import UnconditionedModel


def main():

    # Set up an unconditoned model with a combined random
    # multivariate normal / half normal prior.
    mean = numpy.random.rand(3)

    covariance = numpy.random.rand(3, 3)
    covariance = numpy.dot(covariance, covariance.T)

    prior_settings = {
        ("a", "b", "c"): ["multivariate normal", [mean, covariance]],
        "d": ["half normal", [0.01]],
    }

    initial_parameters = numpy.array([*mean, 0.01])

    # Build the model / models.
    model = UnconditionedModel("multivariate", prior_settings, {})

    # Run the simulation.
    sampler = None  # NUTS(None, ModelCollection(model.name, [model]), 1.0)

    simulation = MCMCSimulation(
        model_collection=model,
        warm_up_steps=100000,
        steps=1500000,
        discard_warm_up_data=True,
        sampler=sampler
    )

    trace, log_p_trace, percent_deviation_trace = simulation.run(initial_parameters)
    model.plot(trace, log_p_trace, percent_deviation_trace, show=True)


if __name__ == "__main__":
    main()

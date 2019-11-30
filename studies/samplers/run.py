#!/usr/bin/env python3

import numpy

from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.kernels.samplers import NUTS, MetropolisSampler
from bayesiantesting.models import ModelCollection
from bayesiantesting.models.continuous import GaussianModel


def main():

    # Build the model.
    priors = {"uniform": ("uniform", numpy.array([-5.0, 5.0]))}
    model = GaussianModel("gaussian", priors, 0.0, 1.0)

    model_collection = ModelCollection("gaussian", [model])

    # Draw the initial parameter values from the model priors.
    initial_parameters = model.sample_priors()

    # Run a simulation using a Metropolis sampler.
    proposal_scales = numpy.array([initial_parameters / 100.0])

    sampler = MetropolisSampler(None, model_collection, proposal_scales)

    simulation = MCMCSimulation(
        model_collection=model,
        warm_up_steps=50000,
        steps=50000,
        discard_warm_up_data=True,
        sampler=sampler,
        output_directory_path="metropolis",
    )

    trace, log_p_trace, percent_deviation_trace = simulation.run(initial_parameters)
    model.plot(trace, log_p_trace, percent_deviation_trace, show=True)

    # Run a simulation using a NUTS sampler.
    def log_p_function(parameters, _):
        return model.evaluate_log_posterior(parameters)

    step_size = 1.0  # NUTS.find_reasonable_epsilon(initial_parameters, log_p_function)
    sampler = NUTS(None, model_collection, step_size)

    simulation = MCMCSimulation(
        model_collection=model,
        warm_up_steps=5000,
        steps=10000,
        discard_warm_up_data=True,
        sampler=sampler,
        output_directory_path="nuts",
    )

    trace, log_p_trace, percent_deviation_trace = simulation.run(initial_parameters)
    model.plot(trace, log_p_trace, percent_deviation_trace, show=True)


if __name__ == "__main__":
    main()

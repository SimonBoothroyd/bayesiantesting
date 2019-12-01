#!/usr/bin/env python3

import numpy

from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.models import ModelCollection
from bayesiantesting.models.continuous import GaussianModel
from bayesiantesting.samplers import NUTS, MetropolisSampler


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
        model_collection=model, initial_parameters=initial_parameters, sampler=sampler,
    )

    trace, log_p_trace, percent_deviation_trace = simulation.run(
        warm_up_steps=50000, steps=50000, output_directory="metropolis",
    )

    model.plot(trace, log_p_trace, percent_deviation_trace, show=True)

    # Run a simulation using a NUTS sampler.
    sampler = NUTS(None, model_collection, step_size=1.0)

    simulation = MCMCSimulation(
        model_collection=model, initial_parameters=initial_parameters, sampler=sampler
    )

    trace, log_p_trace, percent_deviation_trace = simulation.run(
        warm_up_steps=5000, steps=10000, output_directory="nuts",
    )

    model.plot(trace, log_p_trace, percent_deviation_trace, show=True)


if __name__ == "__main__":
    main()

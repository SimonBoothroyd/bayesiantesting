#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:42:37 2019

@author: owenmadin
"""

import autograd.numpy
from matplotlib import pyplot
import numpy
from tqdm import tqdm

from bayesiantesting.kernels.samplers import NUTS
from bayesiantesting.utils.distributions import Normal


def main():

    mu = numpy.array([0.0, 0.0])
    sigma = numpy.array([1.0, 5000.0])

    scale = 1.0 / (3.0 * sigma)

    gaussian = Normal(mu, sigma)

    def log_pdf(parameters):
        return autograd.numpy.sum(gaussian.log_pdf(parameters / scale))

    # Draw the initial parameter values from the model priors.
    initial_parameters = gaussian.sample() * scale

    # Run the simulation.
    step_size = NUTS.find_reasonable_epsilon(initial_parameters,
                                             log_pdf)

    sampler = NUTS(log_pdf,
                   len(mu),
                   step_size)

    trace = []
    step_size = []
    log_p_trace = []

    current_parameters = initial_parameters.copy()

    for step in tqdm(range(5000)):

        current_parameters, _ = sampler.step(current_parameters, step < 1000)
        step_size.append(sampler._step_size)

        trace.append(current_parameters / scale)
        log_p_trace.append(log_pdf(current_parameters))

    step_size = autograd.numpy.asarray(step_size)

    trace = autograd.numpy.asarray(trace)
    log_p_trace = autograd.numpy.asarray(log_p_trace)

    # Plot the output.
    pyplot.plot(step_size)
    pyplot.show()

    for i in range(len(mu)):
        pyplot.plot(trace[:, i])
        pyplot.draw()
        pyplot.show()

        pyplot.hist(trace[:, i])
        pyplot.draw()
        pyplot.show()

    pyplot.plot(log_p_trace)
    pyplot.show()


if __name__ == "__main__":
    main()

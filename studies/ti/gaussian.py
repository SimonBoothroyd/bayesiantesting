#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:42:37 2019

@author: owenmadin
"""

import numpy

# from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.kernels.bayes import ThermodynamicIntegration
from bayesiantesting.models import Model
from bayesiantesting.utils import distributions
from matplotlib import pyplot


class GaussianModel(Model):
    """A representation of the two-center Lennard-Jones model, which
    can be evaluated using a surrogate model against a `NISTDataSet`.
    """

    def __init__(self, name, prior_settings, loc, scale):

        super().__init__(name, prior_settings, {})

        self._loc = loc
        self._scale = scale

    def evaluate_log_likelihood(self, parameters):
        return distributions.Normal(self._loc, self._scale).log_pdf(parameters)

    def compute_percentage_deviations(self, parameters):
        return {}


def main():

    priors = {"uniform": ("uniform", numpy.array([-5.0, 5.0]))}

    # Build the model / models.
    model = GaussianModel("gaussian", priors, 0.0, 1.0)

    # Draw the initial parameter values from the model priors.
    initial_parameters = model.sample_priors()

    # simulation = MCMCSimulation(
    #     model_collection=model,
    #     warm_up_steps=100000,
    #     steps=50000,
    #     discard_warm_up_data=True,
    # )
    #
    # trace, log_p_trace, percent_deviation_trace = simulation.run(initial_parameters)
    #
    # # Plot the output.
    # for i in range(model.n_trainable_parameters):
    #     pyplot.plot(trace[:, i + 1])
    #     pyplot.draw()
    #     pyplot.show()
    #
    #     pyplot.hist(trace[:, i + 1])
    #     pyplot.draw()
    #     pyplot.show()
    #
    # pyplot.plot(log_p_trace)
    # pyplot.show()
    #
    # for property_label in percent_deviation_trace:
    #     pyplot.plot(percent_deviation_trace[property_label], label=property_label)
    #
    # pyplot.legend()
    # pyplot.draw()
    # pyplot.show()

    simulation = ThermodynamicIntegration(
        legendre_gauss_degree=15,
        model=model,
        warm_up_steps=1000000,
        steps=50000,
        discard_warm_up_data=True,
    )

    results, integral = simulation.run(initial_parameters, number_of_threads=15)

    print(f"Final Integral:", integral)
    print("==============================")

    d_log_p_d_lambdas = numpy.zeros(len(results))
    d_log_p_d_lambdas_std = numpy.zeros(len(results))

    for index, result in enumerate(results):

        trace, log_p_trace, d_lop_p_d_lambda = result

        figure, axes = pyplot.subplots(
            nrows=2,
            ncols=max(2, model.n_trainable_parameters),
            dpi=200,
            figsize=(10, 10),
        )

        # Plot the output.
        for i in range(model.n_trainable_parameters):
            axes[0, i].plot(trace[:, i + 1])

        axes[1, 0].plot(log_p_trace)
        axes[1, 1].plot(d_lop_p_d_lambda)

        pyplot.draw()
        pyplot.show()

        d_log_p_d_lambdas[index] = numpy.mean(d_lop_p_d_lambda)
        d_log_p_d_lambdas_std[index] = numpy.std(d_lop_p_d_lambda)

    pyplot.errorbar(
        list(range(len(d_log_p_d_lambdas))),
        d_log_p_d_lambdas,
        yerr=d_log_p_d_lambdas_std,
    )
    pyplot.draw()
    pyplot.show()


if __name__ == "__main__":
    main()

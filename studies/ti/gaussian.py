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

    # Run the simulation
    simulation = ThermodynamicIntegration(
        legendre_gauss_degree=20,
        model=model,
        warm_up_steps=100000,
        steps=500000,
        discard_warm_up_data=True,
        output_directory_path="gaussian",
    )

    _, integral, error = simulation.run(initial_parameters, number_of_threads=20)

    print(f"Final Integral:", integral, " +/- ", error)
    print("==============================")


if __name__ == "__main__":
    main()

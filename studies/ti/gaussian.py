#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:42:37 2019

@author: owenmadin
"""

import numpy

from bayesiantesting.kernels.bayes import ThermodynamicIntegration
from bayesiantesting.models.continuous import GaussianModel


def main():

    priors = {"uniform": ("uniform", numpy.array([-5.0, 5.0]))}

    # Build the model / models.
    model = GaussianModel("gaussian", priors, 0.0, 1.0)

    # Draw the initial parameter values from the model priors.
    initial_parameters = model.sample_priors()

    # Run the simulation
    simulation = ThermodynamicIntegration(
        legendre_gauss_degree=16,
        model=model,
        warm_up_steps=100000,
        steps=500000,
        output_directory_path="gaussian",
    )

    _, integral, error = simulation.run(initial_parameters, number_of_processes=4)

    print("Final Integral:", integral, " +/- ", error)
    print("==============================")


if __name__ == "__main__":
    main()

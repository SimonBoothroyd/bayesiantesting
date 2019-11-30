#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:42:37 2019

@author: owenmadin
"""

import numpy

from bayesiantesting.kernels.bayes import MBARIntegration
from bayesiantesting.models.continuous import CauchyModel, GaussianModel


def main():

    priors = {"uniform": ("uniform", numpy.array([-50.0, 50.0]))}

    # Build the model / models.
    model_a = CauchyModel("cauchy", priors, 0.0, 0.5, 0.75)
    model_b = GaussianModel("gaussian", priors, 0.0, 1.0)

    # Draw the initial parameter values from the model priors.
    initial_parameters = numpy.array([0.5])

    # Set up log spaced lambda windows
    lambda_values = numpy.geomspace(1.0, 2.0, 20) - 1.0

    # Run the simulation
    simulation = MBARIntegration(
        lambda_values=lambda_values,
        model=model_a,
        warm_up_steps=100000,
        steps=1000000,
        discard_warm_up_data=True,
        output_directory_path="gaussian",
        reference_model=model_b,
    )

    _, integral, error = simulation.run(initial_parameters, number_of_processes=20)

    print(
        f"Final Integral:",
        integral,
        " +/- ",
        error,
        f" [{integral - error * 1.96}, {integral + error * 1.96}",
    )

    print("==============================")

    print(f"Expected Integral:", numpy.log(0.75), " +/- ", error)
    print("==============================")


if __name__ == "__main__":
    main()

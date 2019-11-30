#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:42:37 2019

@author: owenmadin
"""

import numpy

from bayesiantesting.kernels.rjmc import BiasedRJMCSimulation
from bayesiantesting.models.discrete import TwoCenterLJModelCollection
from bayesiantesting.utils import distributions
from studies.utilities import get_2clj_model, parse_input_yaml, prepare_data


def main():

    # Load in the simulation parameters.
    simulation_params = parse_input_yaml("basic_run.yaml")

    # Load the data.
    data_set, property_types = prepare_data(simulation_params)

    # Define the initial parameters as pre-computed MAP values,
    maximum_a_posteriori = [
        numpy.array([97.08159, 0.37892, 0.14780]),
        numpy.array([97.29539, 0.37873, 0.14832, 0.01]),
        numpy.array([99.50692, 0.37684]),
    ]

    # Define the mapping distributions
    mapping_distributions = [
        [
            distributions.Normal(97.08159267795193, 1.2637469268027388),
            distributions.Normal(0.3789200872391254, 0.0010927724031321946),
            distributions.Normal(0.14780378661729715, 0.003037464915064883),
        ],
        [
            distributions.Normal(97.2953923060582, 1.378483671271809),
            distributions.Normal(0.37873425430708474, 0.001196249460173617),
            distributions.Normal(0.148316663447079, 0.00330744876139567),
            distributions.HalfNormal(0.03396779721823569),
        ],
        [
            distributions.Normal(99.50692414427614, 0.05847002434279269),
            distributions.Normal(0.3768443138925703, 9.88599085152091e-05),
        ],
    ]

    # Build the model / models.
    sub_models = [
        get_2clj_model("AUA", data_set, property_types, simulation_params),
        get_2clj_model("AUA+Q", data_set, property_types, simulation_params),
        get_2clj_model("UA", data_set, property_types, simulation_params),
    ]

    for mean, model in zip(maximum_a_posteriori, sub_models):
        print(model.evaluate_log_posterior(mean))

    model_collection = TwoCenterLJModelCollection(
        "2CLJ Models", sub_models, mapping_distributions
    )

    # Draw the initial parameter values from the model priors.
    initial_model_index = 1  # torch.randint(len(sub_models), (1,)).item()

    # initial_parameters = generate_initial_parameters(sub_models[initial_model_index])
    initial_parameters = maximum_a_posteriori[initial_model_index]

    bias_factors = simulation_params["biasing_factor"]

    simulation = BiasedRJMCSimulation(
        model_collection=model_collection,
        warm_up_steps=int(simulation_params["steps"] * 0.1),
        steps=simulation_params["steps"],
        discard_warm_up_data=True,
        swap_frequency=simulation_params["swap_freq"],
        log_biases=bias_factors,
    )

    simulation.run(initial_parameters, initial_model_index)

    print("Finished!")


if __name__ == "__main__":
    main()

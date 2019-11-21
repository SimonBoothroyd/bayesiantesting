#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:42:37 2019

@author: owenmadin
"""

import math

import numpy
import torch
import yaml
from bayesiantesting import unit
from bayesiantesting.datasets.nist import NISTDataSet, NISTDataType
from bayesiantesting.kernels.rjmc import BiasedRJMCSimulation
from bayesiantesting.models.continuous import TwoCenterLJModel
from bayesiantesting.models.discrete import TwoCenterLJModelCollection
from bayesiantesting.surrogates import StollWerthSurrogate


def parse_input_yaml(filepath):

    print("Loading simulation params from " + filepath + "...")

    with open(filepath) as file:
        simulation_params = yaml.load(file, Loader=yaml.SafeLoader)

    return simulation_params


def prepare_data(simulation_params):
    """From input parameters, pull appropriate experimental data and
    uncertainty information.
    """

    # Retrieve the constants and thermophysical data
    data_set = NISTDataSet(simulation_params["compound"])

    # Filter the data to selected conditions.
    minimum_temperature = (
        simulation_params["trange"][0]
        * data_set.critical_temperature.value.to(unit.kelvin).magnitude
    )
    maximum_temperature = (
        simulation_params["trange"][1]
        * data_set.critical_temperature.value.to(unit.kelvin).magnitude
    )

    data_set.filter(
        minimum_temperature * unit.kelvin,
        maximum_temperature * unit.kelvin,
        simulation_params["number_data_points"],
    )

    property_types = []

    if simulation_params["properties"] == "All":
        property_types.extend(data_set.data_types)
    else:
        if "rhol" in simulation_params["properties"]:
            property_types.append(NISTDataType.LiquidDensity)
        if "Psat" in simulation_params["properties"]:
            property_types.append(NISTDataType.SaturationPressure)

    return data_set, property_types


def get_model(model_name, data_set, property_types, simulation_params):

    priors = {
        "epsilon": simulation_params["priors"]["epsilon"],
        "sigma": simulation_params["priors"]["sigma"],
    }
    fixed = {}

    if model_name == "AUA":

        priors["L"] = simulation_params["priors"]["L"]
        fixed["Q"] = 0.0

    elif model_name == "AUA+Q":

        priors["L"] = simulation_params["priors"]["L"]
        priors["Q"] = simulation_params["priors"]["Q"]

    elif model_name == "UA":

        fixed["L"] = data_set.bond_length.to(unit.nanometer).magnitude
        fixed["Q"] = 0.0

    else:

        raise NotImplementedError()

    model = TwoCenterLJModel(
        model_name,
        priors,
        fixed,
        data_set,
        property_types,
        StollWerthSurrogate(data_set.molecular_weight),
    )

    return model


def generate_initial_parameters(model):

    initial_log_p = math.nan
    initial_parameters = None

    counter = 0

    while math.isnan(initial_log_p) and counter < 1000:

        initial_parameters = model.sample_priors()
        initial_log_p = model.evaluate_log_posterior(initial_parameters)

        counter += 1

    if numpy.isnan(initial_log_p):

        raise ValueError(
            "The initial values could not be set without yielding "
            "a NaN log posterior"
        )

    return initial_parameters


def main():

    # Load in the simulation parameters.
    simulation_params = parse_input_yaml("basic_run.yaml")

    # Load the data.
    data_set, property_types = prepare_data(simulation_params)

    # Define the pre-computed MAP parameters
    maximum_a_posteriori = [
        numpy.array([97.0, 0.37850, 0.15]),
        numpy.array([98.0, 0.37800, 0.15, 0.01]),
        numpy.array([99.5, 0.37685]),
    ]

    # Build the model / models.
    sub_models = [
        get_model("AUA", data_set, property_types, simulation_params),
        get_model("AUA+Q", data_set, property_types, simulation_params),
        get_model("UA", data_set, property_types, simulation_params),
    ]

    model_collection = TwoCenterLJModelCollection(
        "2CLJ Models", sub_models, maximum_a_posteriori
    )

    # Draw the initial parameter values from the model priors.
    initial_model_index = torch.randint(len(sub_models), (1,)).item()

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

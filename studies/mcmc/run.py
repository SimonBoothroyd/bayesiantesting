#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:42:37 2019

@author: owenmadin
"""
import math

import numpy
import yaml
from bayesiantesting import unit
from bayesiantesting.datasets.nist import NISTDataSet, NISTDataType
from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.models.continuous import TwoCenterLJModel
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


def generate_initial_parameters(model, attempts=5):

    initial_log_p = -math.inf
    initial_parameters = None

    counter = 0

    while counter < attempts:

        parameters = model.sample_priors()
        parameters = model.find_maximum_a_posteriori(parameters)

        log_p = model.evaluate_log_posterior(parameters)
        counter += 1

        if math.isnan(log_p) or log_p < initial_log_p:
            continue

        initial_parameters = parameters
        initial_log_p = log_p

    if numpy.isnan(initial_log_p):

        raise ValueError(
            "The initial values could not be set without yielding "
            "a NaN log posterior"
        )

    return initial_parameters


def main():

    print("Parsing simulation params")
    simulation_params = parse_input_yaml("basic_run.yaml")

    print(simulation_params["priors"])

    # Load the data.
    data_set, property_types = prepare_data(simulation_params)

    # Build the model / models.
    model = get_model("UA", data_set, property_types, simulation_params)

    # Draw the initial parameter values from the model priors.
    # initial_parameters = generate_initial_parameters(model)
    initial_parameters = numpy.array([99.5, 0.37685])

    # Run the simulation.
    simulation = MCMCSimulation(
        model_collection=model,
        warm_up_steps=int(simulation_params["steps"] * 0.3),
        steps=simulation_params["steps"],
        discard_warm_up_data=True,
    )

    trace, log_p_trace, percent_deviation_trace = simulation.run(initial_parameters)
    model.plot(trace, log_p_trace, percent_deviation_trace, show=True)

    print("Finished!")


if __name__ == "__main__":
    main()

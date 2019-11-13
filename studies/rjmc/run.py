#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:42:37 2019

@author: owenmadin
"""

import math
import os

import numpy
import torch
import yaml

from bayesiantesting import unit
from bayesiantesting.datasets.nist import NISTDataSet, NISTDataType
from bayesiantesting.kernels.rjmc import RJMCSimulation
from bayesiantesting.models.continuous import TwoCenterLJModel
from bayesiantesting.models.discrete import TwoCenterLJModelCollection
from bayesiantesting.surrogates import StollWerthSurrogate
from matplotlib import pyplot


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

    # Build the model / models.
    sub_models = [
        get_model("AUA", data_set, property_types, simulation_params),
        get_model("AUA+Q", data_set, property_types, simulation_params),
        get_model("UA", data_set, property_types, simulation_params)
    ]

    model = TwoCenterLJModelCollection('2CLJ Models', sub_models)

    # Draw the initial parameter values from the model priors.
    initial_model_index = torch.randint(len(sub_models), (1,)).item()
    initial_parameters = generate_initial_parameters(sub_models[initial_model_index])

    simulation = RJMCSimulation(
        model=model,
        warm_up_steps=int(simulation_params["steps"] * 0.1),
        steps=simulation_params["steps"],
        discard_warm_up_data=True,
        swap_frequency=simulation_params["swap_freq"],
        optimum_matching=simulation_params["optimum_matching"],
        optimum_bounds="Normal"
    )

    trace, log_p_trace, percent_deviation_trace = simulation.run(initial_parameters, initial_model_index)

    # Plot the output.
    pyplot.plot(log_p_trace)
    pyplot.show()

    # Save the traces to disk.
    os.makedirs("traces", exist_ok=True)

    numpy.save(os.path.join("traces", "trace.npy"), trace)
    numpy.save(os.path.join("traces", "log_p_trace.npy"), log_p_trace)
    numpy.save(os.path.join("traces", "percent_dev_trace.npy"), percent_deviation_trace)

    # rjmc_simulator = rjmc.RJMCSimulation(
    #     simulation_params["compound"],
    #     simulation_params["trange"],
    #     simulation_params["properties"],
    #     simulation_params["number_data_points"],
    #     simulation_params["steps"],
    #     simulation_params["swap_freq"],
    #     simulation_params["biasing_factor"],
    #     simulation_params["optimum_matching"],
    # )
    #
    # rjmc_simulator.prepare_data()
    #
    # print("Simulation Attributes:", rjmc_simulator.get_attributes())
    #
    # compound_2CLJ = surrogates.TwoCenterLJModel(rjmc_simulator.molecular_weight)
    # rjmc_simulator.optimum_bounds = simulation_params["opt_bounds"]
    # rjmc_simulator.gen_Tmatrix(prior, compound_2CLJ)
    # # print(rjmc_simulator.opt_params_AUA)
    # rjmc_simulator.set_initial_state(prior, compound_2CLJ)
    #
    # rjmc_simulator.RJMC_Outerloop(prior, compound_2CLJ)
    # trace, logp_trace, percent_dev_trace, BAR_trace = rjmc_simulator.Report(
    #     USE_BAR=simulation_params["USE_BAR"]
    # )
    #
    # rjmc_simulator.write_output(
    #     simulation_params["priors"],
    #     tag=simulation_params["label"],
    #     save_traj=simulation_params["save_traj"],
    # )
    #
    # path = (
    #     "output/"
    #     + simulation_params["compound"]
    #     + "/"
    #     + simulation_params["properties"]
    #     + "/"
    #     + simulation_params["compound"]
    #     + "_"
    #     + simulation_params["properties"]
    #     + "_"
    #     + str(simulation_params["steps"])
    #     + "_"
    #     + simulation_params["label"]
    #     + "_"
    #     + str(date.today())
    #     + "/runfile.yaml"
    # )
    #
    # with open(path, "w") as outfile:
    #     yaml.dump(simulation_params, outfile, default_flow_style=False)

    print("Finished!")


if __name__ == "__main__":
    main()

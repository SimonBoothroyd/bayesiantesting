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


def generate_initial_values(model):

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

    print("Parsing simulation params")
    simulation_params = parse_input_yaml("basic_run.yaml")

    print(simulation_params["priors"])

    data_set, property_types = prepare_data(simulation_params)

    aua_priors = {
        "epsilon": simulation_params["priors"]["epsilon"],
        "sigma": simulation_params["priors"]["sigma"],
        "L": simulation_params["priors"]["L"]
    }
    aua_fixed = {
        "Q": 0.0
    }

    aua_model = TwoCenterLJModel(
        aua_priors,
        aua_fixed,
        data_set,
        property_types,
        StollWerthSurrogate(data_set.molecular_weight),
    )

    simulation = MCMCSimulation(
        model=aua_model,
        warm_up_steps=int(simulation_params["steps"] * 0.1),
        steps=simulation_params["steps"],
        discard_warm_up_data=False,
    )

    initial_parameters = generate_initial_values(aua_model)

    trace, log_p_trace, percent_deviation_trace = simulation.run(initial_parameters)

    for i in range(4):
        pyplot.plot(trace[:, i])
        pyplot.draw()
        pyplot.show()

    pyplot.plot(log_p_trace)
    pyplot.show()

    for property_label in percent_deviation_trace:
        pyplot.plot(percent_deviation_trace[property_label], label=property_label)

    pyplot.legend()
    pyplot.draw()
    pyplot.show()

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

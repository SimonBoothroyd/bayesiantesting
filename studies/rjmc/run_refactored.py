#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:42:37 2019

@author: owenmadin
"""

import yaml

from bayesiantesting import unit
from bayesiantesting.datasets.nist import NISTDataSet, NISTDataType
from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.models.lj import TwoCenterLennardJones
from bayesiantesting.surrogates import StollWerthSurrogate


def parse_input_yaml(filepath):

    print("Loading simulation params from " + filepath + "...")

    with open(filepath) as file:
        simulation_params = yaml.load(file, Loader=yaml.SafeLoader)

    return simulation_params


def prepare_data(compound, temperature_range, number_of_points):
    """From input parameters, pull appropriate experimental data and
    uncertainty information.
    """

    # Retrieve the constants and thermophysical data
    data_set = NISTDataSet(compound)

    # Filter the data to selected conditions.
    T_min = temperature_range[0] * data_set.critical_temperature.value.to(unit.kelvin).magnitude
    T_max = temperature_range[1] * data_set.critical_temperature.value.to(unit.kelvin).magnitude

    data_set.filter(T_min * unit.kelvin, T_max * unit.kelvin, number_of_points)

    return data_set


def main():

    print("Parsing simulation params")
    simulation_params = parse_input_yaml("basic_run.yaml")

    print(simulation_params["priors"])

    data_set = prepare_data(simulation_params["compound"],
                            simulation_params["trange"],
                            simulation_params["number_data_points"])

    property_types = []

    if "rhol" in simulation_params["properties"]:
        property_types.append(NISTDataType.LiquidDensity)
    elif "Psat" in simulation_params["properties"]:
        property_types.append(NISTDataType.SaturationPressure)
    elif simulation_params["properties"] == "All":
        property_types.extend(data_set.data_types)

    model = TwoCenterLennardJones(simulation_params["priors"],
                                  data_set,
                                  property_types,
                                  StollWerthSurrogate(data_set.molecular_weight))

    simulation = MCMCSimulation(
        model,
        simulation_params["steps"] * 0.1,
        simulation_params["steps"]
    )

    # print("Simulation Attributes:", rjmc_simulator.get_attributes())
    initial_parameters = simulation.set_initial_state()

    trace, log_p_trace = simulation.run(initial_parameters)

    print(log_p_trace)

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

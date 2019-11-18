#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:42:37 2019

@author: owenmadin
"""

import numpy
import yaml
from matplotlib import pyplot

from bayesiantesting import unit
from bayesiantesting.datasets.nist import NISTDataSet, NISTDataType
from bayesiantesting.kernels.bayes import ThermodynamicIntegration
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


def main():

    simulation_params = parse_input_yaml("basic_run.yaml")

    # Load the data.
    data_set, property_types = prepare_data(simulation_params)

    # Build the model / models.
    model = get_model("AUA", data_set, property_types, simulation_params)

    # Draw the initial parameter values from the model priors.
    # initial_parameters = generate_initial_parameters(model)
    initial_parameters = numpy.array([95.0, 0.35, 0.2])

    simulation = ThermodynamicIntegration(
        legendre_gauss_degree=20,
        model=model,
        warm_up_steps=int(simulation_params["steps"] * 0.2),
        steps=simulation_params["steps"],
        discard_warm_up_data=True,
    )

    results, integral = simulation.run(initial_parameters, number_of_threads=20)

    print(f"Final Integral:", integral)
    print("==============================")

    d_log_p_d_lambdas = numpy.zeros(len(results))
    d_log_p_d_lambdas_std = numpy.zeros(len(results))

    for index, result in enumerate(results):

        trace, log_p_trace, d_lop_p_d_lambda = result

        figure, axes = pyplot.subplots(
            nrows=2, ncols=model.n_trainable_parameters, dpi=200, figsize=(10, 10)
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

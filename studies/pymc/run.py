#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:42:37 2019

@author: owenmadin
"""

import pymc3
import theano.tensor as tt
import yaml
from matplotlib import pyplot

from bayesiantesting import unit
from bayesiantesting.datasets.nist import NISTDataSet, NISTDataType
from bayesiantesting.pymc.distributions import StollWerthOp
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


def main():

    print("Parsing simulation params")
    simulation_params = parse_input_yaml("basic_run.yaml")

    # Load the data.
    data_set, property_types = prepare_data(simulation_params)

    # Build the likelihood operator.
    surrogate_model = StollWerthSurrogate(data_set.molecular_weight)
    log_likelihood = StollWerthOp(data_set, property_types, surrogate_model)

    # Define the potentially 'fixed' model constants.
    bond_length = data_set.bond_length.to(unit.nanometer).magnitude
    quadrupole = 0.0

    # Define a value which PyMC3 can use to test that the provided model
    # can be correctly evaluated.
    test_value = tt.as_tensor_variable([95.0, 0.35, bond_length, quadrupole])

    # Build the model / models.
    with pymc3.Model() as model:

        epsilon = pymc3.Bound(pymc3.Exponential, 0.0)("epsilon", lam=1.0 / 400.0)
        sigma = pymc3.Bound(pymc3.Exponential, 0.0)("sigma", lam=1.0 / 5.0)
        bond_length = pymc3.Bound(pymc3.Exponential, 0.0)("bond_length", lam=1.0 / 3.0)

        # Uncomment this line to turn the three parameter model into a two parameter model.
        # quadrupole = pymc3.Bound(pymc3.Exponential, 0.0)('quadrupole', lam=1.0)

        theta = tt.as_tensor_variable([epsilon, sigma, bond_length, quadrupole])

        pymc3.DensityDist(
            "likelihood",
            lambda v: log_likelihood(v),
            observed={"v": theta},
            testval=test_value,
        )

    with model:

        # step = pymc3.NUTS()
        # trace = pymc3.sample(500, step=step)
        trace = pymc3.sample(5000, step=pymc3.Metropolis(), chains=2)

    pymc3.traceplot(trace)
    pyplot.show()

    print("Finished!")


if __name__ == "__main__":
    main()

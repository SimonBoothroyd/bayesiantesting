import pymc3
import theano.tensor as tt

from bayesiantesting import unit
from bayesiantesting.pymc.distributions import StollWerthOp
from bayesiantesting.surrogates import StollWerthSurrogate
from studies.utilities import parse_input_yaml, prepare_data


def build_model(model_name, data_set, property_types):

    # Build the likelihood operator.
    surrogate_model = StollWerthSurrogate(data_set.molecular_weight)
    log_likelihood = StollWerthOp(data_set, property_types, surrogate_model)

    # Define the potentially 'fixed' model constants.
    bond_length = data_set.bond_length.to(unit.nanometer).magnitude
    quadrupole = 0.0

    # Build the model
    with pymc3.Model() as model:

        epsilon = pymc3.Bound(pymc3.Exponential, 0.0)("epsilon", lam=1.0 / 400.0)
        sigma = pymc3.Bound(pymc3.Exponential, 0.0)("sigma", lam=1.0 / 5.0)

        if model_name == "AUA" or model_name == "AUA+Q":
            bond_length = pymc3.Bound(pymc3.Exponential, 0.0)("bond_length", lam=1.0 / 3.0)
        if model_name == "AUA+Q":
            quadrupole = pymc3.Bound(pymc3.Exponential, 0.0)('quadrupole', lam=1.0)

        theta = tt.as_tensor_variable([epsilon, sigma, bond_length, quadrupole])

        pymc3.DensityDist(
            "likelihood",
            lambda v: log_likelihood(v),
            observed={"v": theta},
        )

    return model


def main():

    simulation_parameters = parse_input_yaml("parameters.yaml")

    # Load the training data.
    data_set, property_types = prepare_data(simulation_parameters)

    # Set some initial parameter close to the MAP taken from other runs of C2H6.
    fixed_bond_length = data_set.bond_length.to(unit.nanometer).magnitude

    initial_parameters = {
        "UA": {
            "epsilon": 99.55,
            "sigma": 0.3768,
            "bond_length": fixed_bond_length,
            "quadrupole": 0.0
        },
        "AUA": {
            "epsilon": 140.0,
            "sigma": 0.348,
            "bond_length": 0.243,
            "quadrupole": 0.0
        },
        "AUA+Q": {
            "epsilon": 140.0,
            "sigma": 0.348,
            "bond_length": 0.243,
            "quadrupole": 0.0
        },
    }

    for model_name in ["UA", "AUA", "AUA+Q"]:

        model = build_model(model_name, data_set, property_types)

        with model:

            trace = pymc3.sample(
                draws=simulation_parameters["steps"],
                step=pymc3.Metropolis(),
                chains=2,
                start=initial_parameters[model_name]
            )

        axes = pymc3.traceplot(trace)
        figure = axes[0][0].figure

        figure.savefig(f"{model_name}.png")


if __name__ == "__main__":
    main()

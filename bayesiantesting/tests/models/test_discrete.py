"""
Unit and regression test for the datasets module.
"""
import numpy

from bayesiantesting import unit
from bayesiantesting.datasets.nist import NISTDataSet
from bayesiantesting.models.continuous import TwoCenterLJModel
from bayesiantesting.models.discrete import TwoCenterLJModelCollection
from bayesiantesting.surrogates import StollWerthSurrogate


def get_model(model_name, data_set, property_types):

    priors_settings = {
        "epsilon": ("exponential", numpy.array([0.0, numpy.random.random() * 400.0])),
        "sigma": ("exponential", numpy.array([0.0, numpy.random.random() * 5.0])),
        "L": ("exponential", numpy.array([0.0, numpy.random.random() * 3.0])),
        "Q": ("exponential", numpy.array([0.0, numpy.random.random() * 1.0])),
    }

    priors = {
        "epsilon": priors_settings["epsilon"],
        "sigma": priors_settings["sigma"],
    }
    fixed = {}

    if model_name == "AUA":

        priors["L"] = priors_settings["L"]
        fixed["Q"] = 0.0

    elif model_name == "AUA+Q":

        priors["L"] = priors_settings["L"]
        priors["Q"] = priors_settings["Q"]

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


def test_exponential_priors():

    data_set = NISTDataSet("C2H2")

    sub_models = [
        get_model("AUA", data_set, data_set.data_types),
        get_model("AUA+Q", data_set, data_set.data_types),
        get_model("UA", data_set, data_set.data_types),
    ]

    model = TwoCenterLJModelCollection("2CLJ Models", sub_models)

    # Generate some initial random parameters for each model.
    initial_parameters_0 = sub_models[0].sample_priors()
    initial_parameters_1 = sub_models[1].sample_priors()
    initial_parameters_2 = sub_models[2].sample_priors()

    # Test the case of going to a model with one higher dimension
    original_parameter_0_1, mapped_parameters_0_1, jacobian_0_1 = model.map_parameters(
        initial_parameters_0, 0, 1
    )
    # Test the mapping of epsilon
    assert numpy.isclose(
        jacobian_0_1[0],
        sub_models[0].priors[0].rate.item() / sub_models[1].priors[0].rate.item(),
    )
    # Test the mapping of sigma
    assert numpy.isclose(
        jacobian_0_1[1],
        sub_models[0].priors[1].rate.item() / sub_models[1].priors[1].rate.item(),
    )
    # Test the mapping of L
    assert numpy.isclose(
        jacobian_0_1[2],
        sub_models[0].priors[2].rate.item() / sub_models[1].priors[2].rate.item(),
    )
    # Test the mapping of 'ghost' Q to Q
    assert numpy.isclose(
        jacobian_0_1[3],
        1.0 / (sub_models[1].priors[3].rate.item() * (1.0 - original_parameter_0_1[3])),
    )

    # Test the case of going to a model with one lower dimension
    original_parameter_1_0, mapped_parameters_1_0, jacobian_1_0 = model.map_parameters(
        initial_parameters_1, 1, 0
    )
    # Test the mapping of epsilon
    assert numpy.isclose(
        jacobian_1_0[0],
        sub_models[1].priors[0].rate.item() / sub_models[0].priors[0].rate.item(),
    )
    # Test the mapping of sigma
    assert numpy.isclose(
        jacobian_1_0[1],
        sub_models[1].priors[1].rate.item() / sub_models[0].priors[1].rate.item(),
    )
    # Test the mapping of L
    assert numpy.isclose(
        jacobian_1_0[2],
        sub_models[1].priors[2].rate.item() / sub_models[0].priors[2].rate.item(),
    )
    # Test the mapping of Q to 'ghost' Q
    assert numpy.isclose(
        jacobian_1_0[3],
        sub_models[1].priors[3].rate.item()
        * numpy.exp(-sub_models[1].priors[3].rate.item() * initial_parameters_1[3]),
    )

    # Test the case of going to a model with two lower dimension
    original_parameter_1_2, mapped_parameters_1_2, jacobian_1_2 = model.map_parameters(
        initial_parameters_1, 1, 2
    )
    # Test the mapping of epsilon
    assert numpy.isclose(
        jacobian_1_2[0],
        sub_models[1].priors[0].rate.item() / sub_models[2].priors[0].rate.item(),
    )
    # Test the mapping of sigma
    assert numpy.isclose(
        jacobian_1_2[1],
        sub_models[1].priors[1].rate.item() / sub_models[2].priors[1].rate.item(),
    )
    # Test the mapping of L to 'ghost' L
    assert numpy.isclose(
        jacobian_1_2[2],
        sub_models[1].priors[2].rate.item()
        * numpy.exp(-sub_models[1].priors[2].rate.item() * initial_parameters_1[2]),
    )
    # Test the mapping of Q to 'ghost' Q
    assert numpy.isclose(
        jacobian_1_2[3],
        sub_models[1].priors[3].rate.item()
        * numpy.exp(-sub_models[1].priors[3].rate.item() * initial_parameters_1[3]),
    )

    # Test the case of going to a model with two higher dimension
    original_parameter_2_1, mapped_parameters_2_1, jacobian_2_1 = model.map_parameters(
        initial_parameters_2, 2, 1
    )
    # Test the mapping of epsilon
    assert numpy.isclose(
        jacobian_2_1[0],
        sub_models[2].priors[0].rate.item() / sub_models[1].priors[0].rate.item(),
    )
    # Test the mapping of sigma
    assert numpy.isclose(
        jacobian_2_1[1],
        sub_models[2].priors[1].rate.item() / sub_models[1].priors[1].rate.item(),
    )
    # Test the mapping of L to 'ghost' L
    assert numpy.isclose(
        jacobian_2_1[2],
        1.0 / (sub_models[1].priors[2].rate.item() * (1.0 - original_parameter_2_1[2])),
    )
    # Test the mapping of Q to 'ghost' Q
    assert numpy.isclose(
        jacobian_2_1[3],
        1.0 / (sub_models[1].priors[3].rate.item() * (1.0 - original_parameter_2_1[3])),
    )

    # Test going between models which each have the same fixed value.
    original_parameter_0_2, mapped_parameters_0_2, jacobian_0_2 = model.map_parameters(
        initial_parameters_0, 0, 2
    )
    # Test the mapping of epsilon
    assert numpy.isclose(
        jacobian_0_2[0],
        sub_models[0].priors[0].rate.item() / sub_models[2].priors[0].rate.item(),
    )
    # Test the mapping of sigma
    assert numpy.isclose(
        jacobian_0_2[1],
        sub_models[0].priors[1].rate.item() / sub_models[2].priors[1].rate.item(),
    )
    # Test the mapping of L
    assert numpy.isclose(
        jacobian_0_2[2],
        sub_models[0].priors[2].rate.item()
        * numpy.exp(-sub_models[0].priors[2].rate.item() * initial_parameters_0[2]),
    )
    # Test the mapping of Q to 'ghost' Q
    assert numpy.isclose(jacobian_0_2[3], 1.0)

    original_parameter_2_0, mapped_parameters_2_0, jacobian_2_0 = model.map_parameters(
        initial_parameters_2, 2, 0
    )
    # Test the mapping of epsilon
    assert numpy.isclose(
        jacobian_2_0[0],
        sub_models[2].priors[0].rate.item() / sub_models[0].priors[0].rate.item(),
    )
    # Test the mapping of sigma
    assert numpy.isclose(
        jacobian_2_0[1],
        sub_models[2].priors[1].rate.item() / sub_models[0].priors[1].rate.item(),
    )
    # Test the mapping of L
    assert numpy.isclose(
        jacobian_2_0[2],
        1.0 / (sub_models[0].priors[2].rate.item() * (1.0 - original_parameter_2_0[2])),
    )
    # Test the mapping of Q to 'ghost' Q
    assert numpy.isclose(jacobian_2_0[3], 1.0)

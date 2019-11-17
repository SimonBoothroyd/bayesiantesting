"""
Unit and regression test for the datasets module.
"""
import autograd
import numpy

from random import random

import pytest
from bayesiantesting.datasets.nist import NISTDataSet
from bayesiantesting.models import Model
from bayesiantesting.models.continuous import TwoCenterLJModel
from bayesiantesting.surrogates import StollWerthSurrogate


def _get_two_center_model():

    data_set = NISTDataSet("C2H2")
    property_types = [*data_set.data_types]

    priors = {
        "epsilon": ("exponential", [0.0, random() * 400.0]),
        "sigma": ("exponential", [0.0, random() * 5.0]),
        "L": ("exponential", [0.0, random() * 3.0]),
        "Q": ("exponential", [0.0, random() * 1.0]),
    }
    fixed = {}

    model = TwoCenterLJModel(
        "AUA+Q",
        priors,
        fixed,
        data_set,
        property_types,
        StollWerthSurrogate(data_set.molecular_weight),
    )

    return model


def test_evaluate_log_prior():

    prior_settings = {
        "a": ("exponential", [0.0, random()]),
        "b": ("normal", [0.0, random()]),
    }

    model = Model("test", prior_settings, {})
    parameters = model.sample_priors()

    # Make sure the method call doesn't fail.
    model.evaluate_log_prior(parameters)

    # Test the gradient.
    prior_gradient_function = autograd.grad(model.evaluate_log_prior)
    prior_gradients = prior_gradient_function(parameters)

    assert len(prior_gradients) == len(parameters)
    assert not numpy.allclose(prior_gradients, 0.0)


@pytest.mark.parametrize("model", [_get_two_center_model()])
def test_evaluate_log_likelihood(model):

    parameters = model.sample_priors()

    # Make sure the method call doesn't fail.
    model.evaluate_log_likelihood(parameters)

    # Test the gradient.
    prior_gradient_function = autograd.grad(model.evaluate_log_likelihood)
    prior_gradients = prior_gradient_function(parameters)

    assert len(prior_gradients) == len(parameters)
    assert not numpy.allclose(prior_gradients, 0.0)


@pytest.mark.parametrize("model", [_get_two_center_model()])
def test_evaluate_log_posterior(model):

    parameters = model.sample_priors()

    # Make sure the method call doesn't fail.
    model.evaluate_log_posterior(parameters)

    # Test the gradient.
    prior_gradient_function = autograd.grad(model.evaluate_log_posterior)
    prior_gradients = prior_gradient_function(parameters)

    assert len(prior_gradients) == len(parameters)
    assert not numpy.allclose(prior_gradients, 0.0)


# def test_gradient_speed():
#
#     model = _get_two_center_model()
#
#     mu = numpy.array([94.8, 0.353, 0.120, 0.0])
#     sigma = mu / 50.0
#
#     prior_gradient_function = autograd.grad(model.evaluate_log_posterior)
#     n_iter = 1000
#
#     start = perf_counter_ns()
#
#     for i in range(n_iter):
#         parameters = numpy.random.normal(mu, sigma)
#         prior_gradient_function(parameters)
#
#     stop = perf_counter_ns()
#
#     time = ((stop - start) / n_iter) * 1e-9
#     print(time)

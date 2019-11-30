"""
Unit and regression test for the datasets module.
"""
from random import random

import autograd
import numpy
import pytest
import torch.distributions

from bayesiantesting.datasets.nist import NISTDataSet
from bayesiantesting.models import Model
from bayesiantesting.models.continuous import MultivariateGaussian, TwoCenterLJModel
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

    parameters = None
    log_p = -numpy.inf

    counter = 0

    while numpy.isinf(log_p) and counter < 1000:

        parameters = model.sample_priors()
        # Make sure the method call doesn't fail.
        log_p = model.evaluate_log_prior(parameters)

        counter += 1

    assert not numpy.isinf(log_p)

    # Test the gradient.
    prior_gradient_function = autograd.grad(model.evaluate_log_prior)
    prior_gradients = prior_gradient_function(parameters)

    assert len(prior_gradients) == len(parameters)
    assert not numpy.allclose(prior_gradients, 0.0)


@pytest.mark.parametrize("model", [_get_two_center_model()])
def test_evaluate_log_likelihood(model):

    parameters = None
    log_p = -numpy.inf

    counter = 0

    while numpy.isinf(log_p) and counter < 1000:

        parameters = model.sample_priors()
        # Make sure the method call doesn't fail.
        log_p = model.evaluate_log_likelihood(parameters)

        counter += 1

    assert not numpy.isinf(log_p)

    # Test the gradient.
    prior_gradient_function = autograd.grad(model.evaluate_log_likelihood)
    prior_gradients = prior_gradient_function(parameters)

    assert len(prior_gradients) == len(parameters)
    assert not numpy.allclose(prior_gradients, 0.0)


@pytest.mark.parametrize("model", [_get_two_center_model()])
def test_evaluate_log_posterior(model):

    parameters = None
    log_p = -numpy.inf

    counter = 0

    while numpy.isinf(log_p) and counter < 1000:

        parameters = model.sample_priors()
        # Make sure the method call doesn't fail.
        log_p = model.evaluate_log_posterior(parameters)

        counter += 1

    assert not numpy.isinf(log_p)

    # Test the gradient.
    prior_gradient_function = autograd.grad(model.evaluate_log_posterior)
    prior_gradients = prior_gradient_function(parameters)

    assert len(prior_gradients) == len(parameters)
    assert not numpy.allclose(prior_gradients, 0.0)


def test_multivariate_normal():

    mean_dictionary = {"epsilon": 1.0, "sigma": 5.0}
    mean = numpy.array([*mean_dictionary.values()])

    covariance = numpy.array([[1.0, 3.0 / 5.0], [3.0 / 5.0, 1.0]])

    model = MultivariateGaussian("gaussian", mean_dictionary, covariance)

    sample = model.sample_priors()
    sample_tensor = torch.tensor(sample, requires_grad=True, dtype=torch.float64)

    torch_mean = torch.tensor(mean, requires_grad=False, dtype=torch.float64)
    torch_covariance = torch.tensor(
        covariance, requires_grad=False, dtype=torch.float64
    )

    distribution = torch.distributions.MultivariateNormal(torch_mean, torch_covariance)

    calculated_log_p = model.evaluate_log_prior(sample)
    reference_log_p = distribution.log_prob(sample_tensor).item()

    assert numpy.isclose(calculated_log_p, reference_log_p)

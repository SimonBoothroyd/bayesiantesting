"""
Unit tests for the differentiable distributions
"""
from random import random

import autograd
import numpy

import bayesiantesting.utils.distributions as distributions
import pytest
import torch
import torch.distributions as references


@pytest.mark.parametrize(
    "distribution_type,reference_type,args",
    [
        (distributions.Exponential, references.Exponential, [random()]),
        (distributions.Normal, references.Normal, [random(), random()]),
    ],
)
def test_values(distribution_type, reference_type, args):

    distribution = distribution_type(*args)
    reference = reference_type(*args)

    sample = distribution.sample()
    cdf_sample = distribution.cdf(sample)

    assert numpy.isclose(distribution.cdf(sample), reference.cdf(torch.tensor(sample)))
    assert numpy.isclose(
        distribution.inverse_cdf(cdf_sample), reference.icdf(torch.tensor(cdf_sample))
    )
    assert numpy.isclose(
        distribution.log_pdf(sample), reference.log_prob(torch.tensor(sample))
    )


@pytest.mark.parametrize(
    "distribution_type,reference_type,args",
    [
        (distributions.Exponential, references.Exponential, [random()]),
        (distributions.Normal, references.Normal, [random(), random()]),
    ],
)
def test_gradients(distribution_type, reference_type, args):

    distribution = distribution_type(*args)
    reference = reference_type(*args)

    sample = distribution.sample()
    cdf_sample = distribution.cdf(sample)

    cdf_gradient_function = autograd.grad(distribution.cdf)
    inverse_cdf_gradient_function = autograd.grad(distribution.inverse_cdf)
    log_p_gradient_function = autograd.grad(distribution.log_pdf)

    cdf_gradient = cdf_gradient_function(sample)
    inverse_cdf_gradient = inverse_cdf_gradient_function(cdf_sample)
    log_p_gradient = log_p_gradient_function(sample)

    sample_tensor = torch.tensor(sample, requires_grad=True)
    reference_cdf = reference.cdf(sample_tensor)
    reference_cdf.backward()
    assert numpy.isclose(cdf_gradient, sample_tensor.grad.item())

    sample_tensor = torch.tensor(cdf_sample, requires_grad=True)
    reference_inverse_cdf = reference.icdf(sample_tensor)
    reference_inverse_cdf.backward()
    assert numpy.isclose(inverse_cdf_gradient, sample_tensor.grad.item())

    sample_tensor = torch.tensor(sample, requires_grad=True)
    reference_log_p = reference.log_prob(sample_tensor)
    reference_log_p.backward()
    assert numpy.isclose(log_p_gradient, sample_tensor.grad.item())

"""
Unit and regression test for the datasets module.
"""
import autograd
import numpy

from bayesiantesting import unit
from bayesiantesting.datasets.nist import NISTDataType
from bayesiantesting.surrogates import StollWerthSurrogate


def generate_parameters():
    """Returns a set of parameters (and their reduced values) for
    which regression values for each model property are known.
    """

    epsilon = 98.0
    sigma = 0.37800
    bond_length = 0.15
    quadrupole = 0.01

    quadrupole_star_sqr = (quadrupole * 3.1623) ** 2 / (epsilon * 1.38065 * sigma ** 5)
    bond_length_star = bond_length / sigma

    return (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    )


def test_critical_temperature():

    model = StollWerthSurrogate(30.069 * unit.gram / unit.mole)  # C2H6

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    value = model.critical_temperature(epsilon, sigma, bond_length, quadrupole)
    assert numpy.isclose(value, 310.99575)

    reduced_gradient_function = autograd.grad(model.critical_temperature_star, (0, 1))
    gradient_function = autograd.grad(model.critical_temperature, (0, 1, 2, 3))

    reduced_gradient = reduced_gradient_function(quadrupole_star_sqr, bond_length_star)
    gradient = gradient_function(epsilon, sigma, bond_length, quadrupole)

    assert len(reduced_gradient) == 2 and not numpy.allclose(reduced_gradient, 0.0)
    assert numpy.allclose(
        gradient, numpy.array([3.17342591, 452.59372786, -1140.53645031, 0.00153665])
    )


def test_critical_density():

    model = StollWerthSurrogate(30.069 * unit.gram / unit.mole)  # C2H6

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    reduced_gradient_function = autograd.grad(model.critical_density_star, (0, 1))
    gradient_function = autograd.grad(model.critical_density, (0, 1, 2, 3))

    reduced_gradient = reduced_gradient_function(quadrupole_star_sqr, bond_length_star)
    gradient = gradient_function(epsilon, sigma, bond_length, quadrupole)

    assert len(reduced_gradient) == 2 and not numpy.allclose(reduced_gradient, 0.0)
    assert len(gradient) == 4 and not numpy.allclose(gradient, 0.0)


def test_liquid_density():

    model = StollWerthSurrogate(30.069 * unit.gram / unit.mole)  # C2H6

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    temperatures = numpy.array([308.0])
    temperatures_star = temperatures / epsilon

    value = model.liquid_density(temperatures, epsilon, sigma, bond_length, quadrupole)
    assert numpy.isclose(value, 285.1592692)

    reduced_gradient_function = autograd.grad(model.liquid_density_star, (1, 2))
    gradient_function = autograd.grad(model.liquid_density, (1, 2, 3, 4))

    reduced_gradient = reduced_gradient_function(
        temperatures_star, quadrupole_star_sqr, bond_length_star
    )
    gradient = gradient_function(temperatures, epsilon, sigma, bond_length, quadrupole)

    assert len(reduced_gradient) == 2 and not numpy.allclose(reduced_gradient, 0.0)
    assert numpy.allclose(
        gradient, numpy.array([28.12601669, 2044.99241265, -10856.5686318, 0.01421091])
    )


def test_saturation_pressure():

    model = StollWerthSurrogate(30.069 * unit.gram / unit.mole)  # C2H6

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    temperatures = numpy.array([308.0])
    temperatures_star = temperatures / epsilon

    value = model.saturation_pressure(
        temperatures, epsilon, sigma, bond_length, quadrupole
    )
    assert numpy.isclose(value, 5089.09761408)

    reduced_gradient_function = autograd.grad(model.saturation_pressure_star, (1, 2))
    gradient_function = autograd.grad(model.saturation_pressure, (1, 2, 3, 4))

    reduced_gradient = reduced_gradient_function(
        temperatures_star, quadrupole_star_sqr, bond_length_star
    )
    gradient = gradient_function(temperatures, epsilon, sigma, bond_length, quadrupole)

    assert len(reduced_gradient) == 2 and not numpy.allclose(reduced_gradient, 0.0)
    assert numpy.allclose(
        gradient,
        numpy.array([-235.19495191, -69856.91940789, 74257.50320125, -0.11145016]),
    )


def test_surface_tension():

    model = StollWerthSurrogate(30.069 * unit.gram / unit.mole)  # C2H6

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    temperatures = numpy.array([308.0])
    temperatures_star = temperatures / epsilon

    value = model.surface_tension(temperatures, epsilon, sigma, bond_length, quadrupole)
    assert numpy.isclose(value, 0.00017652)

    reduced_gradient_function = autograd.grad(model.surface_tension_star, (1, 2))
    gradient_function = autograd.grad(model.surface_tension, (1, 2, 3, 4))

    reduced_gradient = reduced_gradient_function(
        temperatures_star, quadrupole_star_sqr, bond_length_star
    )
    gradient = gradient_function(temperatures, epsilon, sigma, bond_length, quadrupole)

    assert len(reduced_gradient) == 2 and not numpy.allclose(reduced_gradient, 0.0)
    assert numpy.allclose(
        gradient, numpy.array([0.00023103, 0.03215253, -0.08337817, 9.44823625e-07])
    )


def test_evaluate():

    model = StollWerthSurrogate(30.069 * unit.gram / unit.mole)  # C2H6
    epsilon, sigma, bond_length, _, quadrupole, _ = generate_parameters()

    parameters = numpy.array([epsilon, sigma, bond_length, quadrupole])
    temperatures = numpy.array([298.0, 300.0, 308.0])

    gradient_function = autograd.jacobian(model.evaluate, 1)

    density_gradients = gradient_function(
        NISTDataType.LiquidDensity, parameters, temperatures
    )
    pressure_gradients = gradient_function(
        NISTDataType.SaturationPressure, parameters, temperatures
    )
    tension_gradients = gradient_function(
        NISTDataType.SurfaceTension, parameters, temperatures
    )

    assert (
        density_gradients.shape == pressure_gradients.shape == tension_gradients.shape
    )

    assert not numpy.allclose(density_gradients, 0.0)
    assert not numpy.allclose(pressure_gradients, 0.0)
    assert not numpy.allclose(tension_gradients, 0.0)

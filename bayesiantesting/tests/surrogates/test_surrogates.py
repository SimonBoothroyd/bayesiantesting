"""
Unit and regression test for the datasets module.
"""
import autograd
import numpy

from bayesiantesting import unit
from bayesiantesting.datasets.nist import NISTDataType
from bayesiantesting.surrogates import StollWerthSurrogate


def generate_parameters():

    # epsilon_distribution = torch.distributions.Exponential(rate=1.0/400.0)
    # sigma_distribution = torch.distributions.Exponential(rate=1.0/5.0)
    # bond_length_distribution = torch.distributions.Exponential(rate=1.0/3.0)
    # quadrupole_distribution = torch.distributions.Exponential(rate=1.0/1.0)
    #
    # epsilon = epsilon_distribution.rsample().item()
    # sigma = sigma_distribution.rsample().item()
    # bond_length = bond_length_distribution.rsample().item()
    # quadrupole = quadrupole_distribution.rsample().item()

    epsilon = 79.89
    sigma = 0.35819
    bond_length = 0.12976000000000001
    quadrupole = 0.508

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


def critical_temperature_gradient_analytical(
    model, quadrupole_star_sqr, bond_length_star
):

    q = quadrupole_star_sqr
    l = bond_length_star

    b = model.critical_temperature_star_parameters

    t_c_star_q = (
        2.0 * q ** 1 * b[1]
        + 3.0 * q ** 2 * b[2]
        + 2.0 * q ** 1 / (0.1 + l ** 2) * b[5]
        + 2.0 * q ** 1 / (0.1 + l ** 5) * b[6]
        + 3.0 * q ** 2 / (0.1 + l ** 2) * b[7]
        + 3.0 * q ** 2 / (0.1 + l ** 5) * b[8]
    )
    t_c_star_l = (
        -2.0 * l / (0.1 + l ** 2) ** 2 * b[3]
        + -5.0 * l ** 4 / (0.1 + l ** 5) ** 2 * b[4]
        + q ** 2 * -2.0 * l / (0.1 + l ** 2) ** 2 * b[5]
        + q ** 2 * -5.0 * l ** 4 / (0.1 + l ** 5) ** 2 * b[6]
        + q ** 3 * -2.0 * l / (0.1 + l ** 2) ** 2 * b[7]
        + q ** 3 * -5.0 * l ** 4 / (0.1 + l ** 5) ** 2 * b[8]
    )

    return numpy.array([t_c_star_q, t_c_star_l])


def test_critical_temperature_gradient():

    model = StollWerthSurrogate(26.038 * unit.gram / unit.mole)  # C2H2

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    t_c_star_function = autograd.grad(model.critical_temperature_star, (0, 1))
    t_c_star_gradient = t_c_star_function(quadrupole_star_sqr, bond_length_star)

    t_c_function = autograd.grad(model.critical_temperature, (0, 1, 2, 3))
    t_c_gradient = t_c_function(epsilon, sigma, bond_length, quadrupole)

    t_c_star_grad_analytical = critical_temperature_gradient_analytical(
        model, quadrupole_star_sqr, bond_length_star
    )

    assert numpy.allclose(t_c_star_gradient, t_c_star_grad_analytical)
    assert numpy.isclose(t_c_gradient[2], t_c_star_grad_analytical[1] * epsilon / sigma)


def test_critical_density_gradient():

    model = StollWerthSurrogate(26.038 * unit.gram / unit.mole)  # C2H2

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    rho_c_star_function = autograd.grad(model.critical_density_star, (0, 1))
    rho_c_star_gradient = rho_c_star_function(quadrupole_star_sqr, bond_length_star)

    rho_c_function = autograd.grad(model.critical_temperature, (0, 1, 2, 3))
    rho_c_gradient = rho_c_function(epsilon, sigma, bond_length, quadrupole)

    # rho_c_star_grad_analytical = critical_density_gradient_analytical(model, quadrupole_star_sqr, bond_length_star)
    # rho_c_grad_analytical = rho_c_star_grad_analytical

    assert len(rho_c_star_gradient) == 2 and not numpy.allclose(
        rho_c_star_gradient, 0.0
    )


def test_liquid_density():

    model = StollWerthSurrogate(26.038 * unit.gram / unit.mole)  # C2H2

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    temperatures = numpy.array([298.0, 308.0])
    temperatures_star = temperatures / epsilon

    model.liquid_density_star(temperatures_star, quadrupole_star_sqr, bond_length_star)

    # We have to build a separate gradient for each parameter due the
    # autograd jacobian method not supporting tuples for the argument
    # number. This won't be an issue if the change is made to JAX.
    bond_length_gradient_function = autograd.jacobian(model.liquid_density_star, 1)
    quadrupole_gradient_function = autograd.jacobian(model.liquid_density_star, 2)

    bond_length_gradient = bond_length_gradient_function(
        temperatures_star, quadrupole_star_sqr, bond_length_star
    )
    quadrupole_gradient = quadrupole_gradient_function(
        temperatures_star, quadrupole_star_sqr, bond_length_star
    )

    # <class 'tuple'>: (array(-0.07604066), array(1.6157372))
    # <class 'tuple'>: (array(-0.08812341), array(1.91128129))
    # <class 'tuple'>: (array(-0.10083591), array(2.23236391))

    assert len(bond_length_gradient) == 2 and not numpy.allclose(
        bond_length_gradient, 0.0
    )
    assert len(quadrupole_gradient) == 2 and not numpy.allclose(
        bond_length_gradient, 0.0
    )


def test_saturation_pressure():

    model = StollWerthSurrogate(26.038 * unit.gram / unit.mole)  # C2H2

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    temperatures = numpy.array([298.0, 308.0, 318.0])
    temperatures_star = temperatures / epsilon

    # We have to build a separate gradient for each parameter due the
    # autograd jacobian method not supporting tuples for the argument
    # number. This won't be an issue if the change is made to JAX.
    bond_length_gradient_function = autograd.jacobian(model.saturation_pressure_star, 1)
    quadrupole_gradient_function = autograd.jacobian(model.saturation_pressure_star, 2)

    bond_length_gradient = bond_length_gradient_function(
        temperatures_star, quadrupole_star_sqr, bond_length_star
    )
    quadrupole_gradient = quadrupole_gradient_function(
        temperatures_star, quadrupole_star_sqr, bond_length_star
    )

    # <class 'tuple'>: (array(-0.07604066), array(1.6157372))
    # <class 'tuple'>: (array(-0.08812341), array(1.91128129))
    # <class 'tuple'>: (array(-0.10083591), array(2.23236391))

    assert len(bond_length_gradient) == 3 and not numpy.allclose(
        bond_length_gradient, 0.0
    )
    assert len(quadrupole_gradient) == 3 and not numpy.allclose(
        bond_length_gradient, 0.0
    )

    epsilon_gradient_function = autograd.jacobian(model.saturation_pressure, 1)
    sigma_gradient_function = autograd.jacobian(model.saturation_pressure, 2)
    bond_length_gradient_function = autograd.jacobian(model.saturation_pressure, 3)
    quadrupole_gradient_function = autograd.jacobian(model.saturation_pressure, 4)

    epsilon_gradient = epsilon_gradient_function(
        temperatures, epsilon, sigma, bond_length, quadrupole
    )
    sigma_gradient = sigma_gradient_function(
        temperatures, epsilon, sigma, bond_length, quadrupole
    )
    bond_length_gradient = bond_length_gradient_function(
        temperatures, epsilon, sigma, bond_length, quadrupole
    )
    quadrupole_gradient = quadrupole_gradient_function(
        temperatures, epsilon, sigma, bond_length, quadrupole
    )

    assert len(epsilon_gradient) == 3 and not numpy.allclose(epsilon_gradient, 0.0)
    assert len(sigma_gradient) == 3 and not numpy.allclose(sigma_gradient, 0.0)
    assert len(bond_length_gradient) == 3 and not numpy.allclose(
        bond_length_gradient, 0.0
    )
    assert len(quadrupole_gradient) == 3 and not numpy.allclose(
        bond_length_gradient, 0.0
    )


def test_surface_tension():

    model = StollWerthSurrogate(26.038 * unit.gram / unit.mole)  # C2H2

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    temperatures = numpy.array([298.0, 308.0, 318.0])
    temperatures_star = temperatures / epsilon

    # We have to build a separate gradient for each parameter due the
    # autograd jacobian method not supporting tuples for the argument
    # number. This won't be an issue if the change is made to JAX.
    bond_length_gradient_function = autograd.jacobian(model.surface_tension_star, 1)
    quadrupole_gradient_function = autograd.jacobian(model.surface_tension_star, 2)

    bond_length_gradient = bond_length_gradient_function(
        temperatures_star, quadrupole_star_sqr, bond_length_star
    )
    quadrupole_gradient = quadrupole_gradient_function(
        temperatures_star, quadrupole_star_sqr, bond_length_star
    )

    assert len(bond_length_gradient) == 3 and not numpy.allclose(
        bond_length_gradient, 0.0
    )
    assert len(quadrupole_gradient) == 3 and not numpy.allclose(
        bond_length_gradient, 0.0
    )

    epsilon_gradient_function = autograd.jacobian(model.surface_tension, 1)
    sigma_gradient_function = autograd.jacobian(model.surface_tension, 2)
    bond_length_gradient_function = autograd.jacobian(model.surface_tension, 3)
    quadrupole_gradient_function = autograd.jacobian(model.surface_tension, 4)

    epsilon_gradient = epsilon_gradient_function(
        temperatures, epsilon, sigma, bond_length, quadrupole
    )
    sigma_gradient = sigma_gradient_function(
        temperatures, epsilon, sigma, bond_length, quadrupole
    )
    bond_length_gradient = bond_length_gradient_function(
        temperatures, epsilon, sigma, bond_length, quadrupole
    )
    quadrupole_gradient = quadrupole_gradient_function(
        temperatures, epsilon, sigma, bond_length, quadrupole
    )

    assert len(epsilon_gradient) == 3 and not numpy.allclose(epsilon_gradient, 0.0)
    assert len(sigma_gradient) == 3 and not numpy.allclose(sigma_gradient, 0.0)
    assert len(bond_length_gradient) == 3 and not numpy.allclose(
        bond_length_gradient, 0.0
    )
    assert len(quadrupole_gradient) == 3 and not numpy.allclose(
        bond_length_gradient, 0.0
    )


def test_evaluate():

    model = StollWerthSurrogate(26.038 * unit.gram / unit.mole)  # C2H2
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

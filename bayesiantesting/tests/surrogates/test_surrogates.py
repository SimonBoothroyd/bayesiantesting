"""
Unit and regression test for the datasets module.
"""
import autograd
import numpy

from bayesiantesting import unit
from bayesiantesting.datasets.nist import NISTDataType
from bayesiantesting.surrogates import StollWerthSurrogate
from bayesiantesting.surrogates.surrogates import D_to_sqrtJm3, m3_to_nm3, k_B


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

    quadrupole_sqr = (quadrupole * D_to_sqrtJm3) ** 2 * m3_to_nm3
    quadrupole_star_sqr = quadrupole_sqr / (epsilon * k_B * sigma ** 5)

    bond_length_star = bond_length / sigma

    return (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    )


def dummy_function_separate(values, x_0, x_1):
    return values * (x_0 ** 2 + x_1 ** 3)


def dummy_function_array(values, x):
    return values * (x[0] ** 2 + x[1] ** 3)


def test_autograd():
    """A simple check to make sure my understanding of the
    autograd syntax is correct.

    Returns
    -------

    """

    gradient_function_separate = autograd.grad(dummy_function_separate, (1, 2))
    gradient_function_array = autograd.grad(dummy_function_array, 1)

    x_0 = 3.0
    x_1 = 2.0

    x = numpy.array([x_0, x_1])

    values = 1.0

    gradient_separate = gradient_function_separate(values, x_0, x_1)
    gradient_array = gradient_function_array(values, x)

    assert numpy.allclose(gradient_separate, gradient_array)

    values = numpy.array([1.0, 0.5, 0.25])

    jacobian_function_separate_0 = autograd.jacobian(dummy_function_separate, 1)
    jacobian_function_separate_1 = autograd.jacobian(dummy_function_separate, 2)

    jacobian_function_array = autograd.jacobian(dummy_function_array, 1)

    jacobian_separate = numpy.array(
        [
            jacobian_function_separate_0(values, x_0, x_1),
            jacobian_function_separate_1(values, x_0, x_1),
        ]
    ).T

    jacobian_array = jacobian_function_array(values, x)

    assert numpy.allclose(jacobian_separate, jacobian_array)


def critical_temperature_gradient_analytical(
    model, quadrupole_star_sqr, bond_length_star
):

    b = model.critical_temperature_star_parameters

    t_c_star_q = (
        2.0 * quadrupole_star_sqr ** 1 * b[1]
        + 3.0 * quadrupole_star_sqr ** 2 * b[2]
        + 2.0 * quadrupole_star_sqr ** 1 / (0.1 + bond_length_star ** 2) * b[5]
        + 2.0 * quadrupole_star_sqr ** 1 / (0.1 + bond_length_star ** 5) * b[6]
        + 3.0 * quadrupole_star_sqr ** 2 / (0.1 + bond_length_star ** 2) * b[7]
        + 3.0 * quadrupole_star_sqr ** 2 / (0.1 + bond_length_star ** 5) * b[8]
    )

    t_c_star_l = (
        -2.0 * bond_length_star / (0.1 + bond_length_star ** 2) ** 2 * b[3]
        + -5.0 * bond_length_star ** 4 / (0.1 + bond_length_star ** 5) ** 2 * b[4]
        + quadrupole_star_sqr ** 2
        * -2.0
        * bond_length_star
        / (0.1 + bond_length_star ** 2) ** 2
        * b[5]
        + quadrupole_star_sqr ** 2
        * -5.0
        * bond_length_star ** 4
        / (0.1 + bond_length_star ** 5) ** 2
        * b[6]
        + quadrupole_star_sqr ** 3
        * -2.0
        * bond_length_star
        / (0.1 + bond_length_star ** 2) ** 2
        * b[7]
        + quadrupole_star_sqr ** 3
        * -5.0
        * bond_length_star ** 4
        / (0.1 + bond_length_star ** 5) ** 2
        * b[8]
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

    # rho_c_star_grad_analytical = critical_density_gradient_analytical(model, quadrupole_star_sqr, bond_length_star)
    # rho_c_grad_analytical = rho_c_star_grad_analytical

    assert len(rho_c_star_gradient) == 2 and not numpy.allclose(
        rho_c_star_gradient, 0.0
    )


def test_density():

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

    model.density_star(
        temperatures_star, quadrupole_star_sqr, bond_length_star, "liquid"
    )
    model.density_star(
        temperatures_star, quadrupole_star_sqr, bond_length_star, "vapor"
    )

    # We have to build a separate gradient for each parameter due the
    # autograd jacobian method not supporting tuples for the argument
    # number. This won't be an issue if the change is made to JAX.
    bond_length_gradient_function = autograd.jacobian(model.density_star, 1)
    quadrupole_gradient_function = autograd.jacobian(model.density_star, 2)

    for phase in ["liquid", "vapor"]:

        bond_length_gradient = bond_length_gradient_function(
            temperatures_star, quadrupole_star_sqr, bond_length_star, phase
        )
        quadrupole_gradient = quadrupole_gradient_function(
            temperatures_star, quadrupole_star_sqr, bond_length_star, phase
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

        # epsilon_gradient_function = autograd.jacobian(model.saturation_pressure, 1)
        # sigma_gradient_function = autograd.jacobian(model.saturation_pressure, 2)
        # bond_length_gradient_function = autograd.jacobian(model.saturation_pressure, 3)
        # quadrupole_gradient_function = autograd.jacobian(model.saturation_pressure, 4)
        #
        # epsilon_gradient = epsilon_gradient_function(temperatures, epsilon, sigma, bond_length, quadrupole)
        # sigma_gradient = sigma_gradient_function(temperatures, epsilon, sigma, bond_length, quadrupole)
        # bond_length_gradient = bond_length_gradient_function(temperatures, epsilon, sigma, bond_length, quadrupole)
        # quadrupole_gradient = quadrupole_gradient_function(temperatures, epsilon, sigma, bond_length, quadrupole)
        #
        # assert len(epsilon_gradient) == 2 and not numpy.allclose(epsilon_gradient, 0.0)
        # assert len(sigma_gradient) == 2 and not numpy.allclose(sigma_gradient, 0.0)
        # assert len(bond_length_gradient) == 2 and not numpy.allclose(bond_length_gradient, 0.0)
        # assert len(quadrupole_gradient) == 2 and not numpy.allclose(bond_length_gradient, 0.0)


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

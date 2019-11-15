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


def _sum_property_values(parameters, model, property_types, temperatures):

    total_value = 0.0

    for property_type in property_types:
        total_value += model.evaluate(property_type, parameters, temperatures)

    return total_value


def test_for_loops():

    model = StollWerthSurrogate(26.038 * unit.gram / unit.mole)  # C2H2
    epsilon, sigma, bond_length, _, quadrupole, _ = generate_parameters()

    property_types = [
        NISTDataType.LiquidDensity,
        NISTDataType.SaturationPressure,
        NISTDataType.SurfaceTension
    ]

    parameters = numpy.array([epsilon, sigma, bond_length, quadrupole])
    temperatures = numpy.array([298.0, 300.0, 308.0])

    gradient_function = autograd.jacobian(_sum_property_values, 0)
    gradient = gradient_function(parameters, model, property_types, temperatures)

    print(gradient)

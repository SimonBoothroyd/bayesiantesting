import abc
import os

from autograd import numpy as np
import yaml
from pkg_resources import resource_filename

from bayesiantesting.datasets.nist import NISTDataType


class SurrogateModel(abc.ABC):
    """The base representation of a surrogate model which can
    be cheaply evaluated.
    """

    def evaluate(self, property_type, parameters, temperatures):
        """Evaluate this model for a set of parameters.

        Parameters
        ----------
        property_type: NISTDataType
            The property to evaluate.
        parameters: numpy.ndarray
            The values of the parameters to evaluate at with
            shape=(n parameters, 1).
        temperatures: numpy.ndarray
            The temperatures to evaluate the properties at with
            shape=(n_temperatures).

        Returns
        -------
        numpy.ndarray
            The values of this model at these parameters.
        """
        raise NotImplementedError()


class StollWerthSurrogate(SurrogateModel):
    """A surrogate model for the two-center Lennard-Jones model, which can
    be rapidly evaluated from the models critical density and temperature,
    liquid and vapor density, saturation pressure and surface tension.
    """

    _reduced_boltzmann = 1.38065
    _reduced_D_to_sqrt_J_m3 = 3.1623

    def __init__(self, molecular_weight, file_path=None):
        """Constructs a new `StollWerthSurrogate` object.

        Parameters
        ----------
        molecular_weight
        file_path: str, optional
            The path to the model parameters. If unset, the built in
            `DCLJQ.yaml` parameters will be used.
        """

        self.molecular_weight = molecular_weight

        if file_path is None:

            file_path = resource_filename(
                "bayesiantesting", os.path.join("data", "models", "DCLJQ.yaml")
            )

        with open(file_path) as file:

            parameters = yaml.load(file, Loader=yaml.SafeLoader)[
                "correlation_parameters"
            ]

        self.critical_temperature_star_parameters = np.array(
            parameters["Stoll"]["T_c_star_params"]
        )
        self.density_star_parameters = np.array(
            parameters["Stoll"]["rho_c_star_params"]
        )

        self._b_C1 = np.array(parameters["Stoll"]["rho_L_star_params"]["C1_params"])
        self._b_C2_L = np.array(parameters["Stoll"]["rho_L_star_params"]["C2_params"])
        self._b_C3_L = np.array(parameters["Stoll"]["rho_L_star_params"]["C3_params"])
        self._b_C2_v = np.array(parameters["Stoll"]["rho_v_star_params"]["C2_params"])
        self._b_C3_v = np.array(parameters["Stoll"]["rho_v_star_params"]["C3_params"])

        self._b_c1 = np.array(parameters["Stoll"]["P_v_star_params"]["c1_params"])
        self._b_c2 = np.array(parameters["Stoll"]["P_v_star_params"]["c2_params"])
        self._b_c3 = np.array(parameters["Stoll"]["P_v_star_params"]["c3_params"])

        self._A_a = np.array(parameters["Werth"]["A_star_params"]["a_params"])
        self._A_b = np.array(parameters["Werth"]["A_star_params"]["b_params"])
        self._A_c = np.array(parameters["Werth"]["A_star_params"]["c_params"])
        self._A_d = np.array(parameters["Werth"]["A_star_params"]["d_params"])
        self._A_e = np.array(parameters["Werth"]["A_star_params"]["e_params"])

        self._B = np.array(parameters["Werth"]["A_star_params"]["B_params"])

    @staticmethod
    def _correlation_function_1(quadrupole_star, bond_length_star, b):

        q = quadrupole_star
        l = bond_length_star

        result = (
            1 * b[0]
            + q ** 2 * b[1]
            + q ** 3 * b[2]
            + l ** 3 / (l + 0.4) ** 3 * b[3]
            + l ** 4 / (l + 0.4) ** 5 * b[4]
            + q ** 2 * l ** 2 / (l + 0.4) * b[5]
            + q ** 2 * l ** 3 / (l + 0.4) ** 7 * b[6]
            + q ** 3 * l ** 2 / (l + 0.4) * b[7]
            + q ** 3 * l ** 3 / (l + 0.4) ** 7 * b[8]
        )

        return result

    @staticmethod
    def _correlation_function_2(quadrupole_star, bond_length_star, b):

        q = quadrupole_star
        l = bond_length_star

        result = (
            1 * b[0]
            + q ** 2 * b[1]
            + q ** 3 * b[2]
            + l ** 2 * b[3]
            + l ** 3 * b[4]
            + q ** 2 * l ** 2 * b[5]
            + q ** 2 * l ** 3 * b[6]
            + q ** 3 * l ** 2 * b[7]
        )

        return result

    @staticmethod
    def _correlation_function_3(quadrupole_star, bond_length_star, b):

        q = quadrupole_star
        l = bond_length_star

        result = (
            1 * b[0]
            + q ** 2 * b[1]
            + q ** 3 * b[2]
            + l * b[3]
            + l ** 4 * b[4]
            + q ** 2 * l * b[5]
            + q ** 2 * l ** 4 * b[6]
            + q ** 3 * l ** 4 * b[7]
        )

        return result

    def _a_correlation_function(self, quadrupole_star, bond_length_star):

        c_a, c_b, c_c, c_d, c_e = self._A_a, self._A_b, self._A_c, self._A_d, self._A_e

        a = 1.0 * c_a
        b = (
            quadrupole_star * c_b[0]
            + quadrupole_star ** 2.0 * c_b[1]
            + quadrupole_star ** 3.0 * c_b[2]
        )
        c = 1.0 / (bond_length_star ** 2.0 + 0.1) * c_c[0]
        d = (
            quadrupole_star ** 2.0 * bond_length_star ** 2.0 * c_d[0]
            + quadrupole_star ** 2.0 * bond_length_star ** 3.0 * c_d[1]
        )
        e = (
            quadrupole_star ** 2 / (bond_length_star ** 2.0 + 0.1) * c_e[0]
            + quadrupole_star ** 2.0 / (bond_length_star ** 5.0 + 0.1) * c_e[1]
        )

        return a + b + c + d + e

    def critical_temperature_star(self, quadrupole_star_sqr, bond_length_star):
        """Computes the reduced critical temperature of the two-center
        Lennard-Jones model for a given set of model parameters.

        Parameters
        ----------
        quadrupole_star_sqr: float
            The reduced quadrupole parameter squared.
        bond_length_star: float
            The reduced bond-length parameter

        Returns
        -------
        float
            The reduced critical temperature.
        """

        q = quadrupole_star_sqr
        l = bond_length_star

        b = self.critical_temperature_star_parameters

        t_c_star = (
            1 * b[0]
            + q ** 2 * b[1]
            + q ** 3 * b[2]
            + 1.0 / (0.1 + l ** 2) * b[3]
            + 1.0 / (0.1 + l ** 5) * b[4]
            + q ** 2 / (0.1 + l ** 2) * b[5]
            + q ** 2 / (0.1 + l ** 5) * b[6]
            + q ** 3 / (0.1 + l ** 2) * b[7]
            + q ** 3 / (0.1 + l ** 5) * b[8]
        )

        return t_c_star

    def critical_temperature(self, epsilon, sigma, bond_length, quadrupole):
        """Computes the critical temperature of the two-center
        Lennard-Jones model for a given set of model parameters.

        Parameters
        ----------
        epsilon: float
            The epsilon parameter in units of K.
        sigma: float
            The sigma parameter in units of nm.
        bond_length: float
            The bond-length parameter in units of nm.
        quadrupole: float
            The quadrupole parameter in units of Debye * nm.

        Returns
        -------
        float
            The critical temperature in units of K.
        """
        quadrupole_star_sqr = (quadrupole * self._reduced_D_to_sqrt_J_m3) ** 2 / (
            epsilon * self._reduced_boltzmann * sigma ** 5
        )
        bond_length_star = bond_length / sigma

        critical_temperature_star = self.critical_temperature_star(
            quadrupole_star_sqr, bond_length_star
        )
        critical_temperature = critical_temperature_star * epsilon

        return critical_temperature

    def critical_density_star(self, quadrupole_star, bond_length_star):
        """Computes the reduced critical density of the two-center
        Lennard-Jones model for a given set of model parameters.

        Parameters
        ----------
        quadrupole_star: float
            The reduced quadrupole parameter.
        bond_length_star: float
            The reduced bond-length parameter

        Returns
        -------
        float
            The reduced critical density.
        """

        q = quadrupole_star
        l = bond_length_star

        b = self.density_star_parameters

        rho_c_star = (
            1 * b[0]
            + q ** 2 * b[1]
            + q ** 3 * b[2]
            + l ** 2 / (0.11 + l ** 2) * b[3]
            + l ** 5 / (0.11 + l ** 5) * b[4]
            + l ** 2 * q ** 2 / (0.11 + l ** 2) * b[5]
            + l ** 5 * q ** 2 / (0.11 + l ** 5) * b[6]
            + l ** 2 * q ** 3 / (0.11 + l ** 2) * b[7]
            + l ** 5 * q ** 3 / (0.11 + l ** 5) * b[8]
        )

        return rho_c_star

    def critical_density(self, epsilon, sigma, bond_length, quadrupole):
        """Computes the critical density of the two-center Lennard-Jones
        model for a given set of model parameters.

        Parameters
        ----------
        epsilon: float
            The epsilon parameter in units of K.
        sigma: float
            The sigma parameter in units of nm.
        bond_length: float
            The bond-length parameter in units of nm.
        quadrupole: float
            The quadrupole parameter in units of Debye * nm.

        Returns
        -------
        numpy.ndarray
            The evaluated densities in units of kg / m3.
        """

        molecular_weight = self.molecular_weight.magnitude

        quadrupole_star_sqr = (quadrupole * self._reduced_D_to_sqrt_J_m3) ** 2 / (
            epsilon * self._reduced_boltzmann * sigma ** 5
        )
        bond_length_star = bond_length / sigma

        rho_star = self.critical_density_star(quadrupole_star_sqr, bond_length_star)
        rho = rho_star * molecular_weight / sigma ** 3 / 6.02214 * 10.0
        return rho  # [kg/m3]

    def liquid_density_star(self, temperature_star, quadrupole_star, bond_length_star):
        """Computes the reduced liquid density of the two-center
        Lennard-Jones model for a given set of model parameters over
        a specified range of temperatures.

        Parameters
        ----------
        temperature_star: numpy.ndarray
            The reduced temperatures to evaluate the reduced density at.
        quadrupole_star: float
            The reduced quadrupole parameter.
        bond_length_star: float
            The reduced bond-length parameter

        Returns
        -------
        numpy.ndarray
            The reduced density.
        """

        _b_C1, _b_C2, _b_C3 = (
            self._b_C1,
            self._b_C2_L,
            self._b_C3_L,
        )

        t_c_star = self.critical_temperature_star(quadrupole_star, bond_length_star)
        rho_c_star = self.critical_density_star(quadrupole_star, bond_length_star)

        tau = t_c_star - temperature_star

        if np.all(tau > 0):

            coefficient_1 = self._correlation_function_1(
                quadrupole_star, bond_length_star, _b_C1
            )
            coefficient_2 = self._correlation_function_2(
                quadrupole_star, bond_length_star, _b_C2
            )
            coefficient_3 = self._correlation_function_3(
                quadrupole_star, bond_length_star, _b_C3
            )

            x_0 = 1.0 * rho_c_star
            x_1 = tau ** (1.0 / 3.0) * coefficient_1
            x_2 = tau * coefficient_2
            x_3 = tau ** (3.0 / 2.0) * coefficient_3

            rho_star = x_0 + x_1 + x_2 + x_3

        else:
            rho_star = np.empty(temperature_star.shape) * np.nan

        return rho_star

    def liquid_density(self, temperature, epsilon, sigma, bond_length, quadrupole):
        """Computes the liquid density of the two-center Lennard-Jones
        model for a given set of model parameters over a specified range
        of temperatures.

        Parameters
        ----------
        temperature: numpy.ndarray
            The temperatures to evaluate the density at in units of K.
        epsilon: float
            The epsilon parameter in units of K.
        sigma: float
            The sigma parameter in units of nm.
        bond_length: float
            The bond-length parameter in units of nm.
        quadrupole: float
            The quadrupole parameter in units of Debye * nm.

        Returns
        -------
        numpy.ndarray
            The evaluated densities in units of kg / m3.
        """

        molecular_weight = self.molecular_weight.magnitude

        # Note that epsilon is defined as epsilon/kB
        temperature_star = temperature / epsilon

        quadrupole_star_sqr = (quadrupole * self._reduced_D_to_sqrt_J_m3) ** 2 / (
            epsilon * self._reduced_boltzmann * sigma ** 5
        )
        bond_length_star = bond_length / sigma

        rho_star = self.liquid_density_star(
            temperature_star, quadrupole_star_sqr, bond_length_star
        )
        rho = rho_star * molecular_weight / sigma ** 3 / 6.02214 * 10.0
        return rho  # [kg/m3]

    def saturation_pressure_star(
        self, temperature_star, quadrupole_star, bond_length_star
    ):
        """Computes the reduced saturation pressure of the two-center
        Lennard-Jones model for a given set of model parameters over
        a specified range of temperatures.

        Parameters
        ----------
        temperature_star: numpy.ndarray
            The reduced temperatures to evaluate the reduced density at.
        quadrupole_star: float
            The reduced quadrupole parameter.
        bond_length_star: float
            The reduced bond-length parameter

        Returns
        -------
        numpy.ndarray
            The reduced saturation pressures.
        """

        _b_c1, _b_c2, _b_c3 = self._b_c1, self._b_c2, self._b_c3

        q = quadrupole_star
        l = bond_length_star

        c1 = (
            1.0 * _b_c1[0]
            + q ** 2 * _b_c1[1]
            + q ** 3 * _b_c1[2]
            + l ** 2 / (l ** 2 + 0.75) * _b_c1[3]
            + l ** 3 / (l ** 3 + 0.75) * _b_c1[4]
            + l ** 2 * q ** 2 / (l ** 2 + 0.75) * _b_c1[5]
            + l ** 3 * q ** 2 / (l ** 3 + 0.75) * _b_c1[6]
            + l ** 2 * q ** 3 / (l ** 2 + 0.75) * _b_c1[7]
            + l ** 3 * q ** 3 / (l ** 3 + 0.75) * _b_c1[8]
        )
        c2 = (
            1.0 * _b_c2[0]
            + q ** 2 * _b_c2[1]
            + q ** 3 * _b_c2[2]
            + l ** 2 / (l + 0.75) ** 2 * _b_c2[3]
            + l ** 3 / (l + 0.75) ** 3 * _b_c2[4]
            + l ** 2 * q ** 2 / (l + 0.75) ** 2 * _b_c2[5]
            + l ** 3 * q ** 2 / (l + 0.75) ** 3 * _b_c2[6]
            + l ** 2 * q ** 3 / (l + 0.75) ** 2 * _b_c2[7]
            + l ** 3 * q ** 3 / (l + 0.75) ** 3 * _b_c2[8]
        )
        c3 = q ** 2 * _b_c3[0] + q ** 5 * _b_c3[1] + l ** 0.5

        saturation_pressure_star = np.exp(
            c1 + c2 / temperature_star + c3 / (temperature_star ** 4)
        )
        return saturation_pressure_star

    def saturation_pressure(self, temperature, epsilon, sigma, bond_length, quadrupole):
        """Computes the saturation pressure of the two-center Lennard-Jones model
        for a given set of model parameters over a specified range of
        temperatures.

        Parameters
        ----------
        temperature: numpy.ndarray
            The temperatures to evaluate the density at in units of K.
        epsilon: float
            The epsilon parameter in units of K.
        sigma: float
            The sigma parameter in units of nm.
        bond_length: float
            The bond-length parameter in units of nm.
        quadrupole: float
            The quadrupole parameter in units of Debye * nm.

        Returns
        -------
        numpy.ndarray
            The evaluated saturation pressures in units of kPa
        """

        temperature_star = temperature / epsilon

        quadrupole_star_sqr = (quadrupole * self._reduced_D_to_sqrt_J_m3) ** 2 / (
            epsilon * self._reduced_boltzmann * sigma ** 5
        )
        bond_length_star = bond_length / sigma

        saturation_pressure_star = self.saturation_pressure_star(
            temperature_star, quadrupole_star_sqr, bond_length_star
        )
        saturation_pressure = (
            saturation_pressure_star
            * epsilon
            / sigma ** 3
            * self._reduced_boltzmann
            * 1.0e1
        )
        return saturation_pressure  # [kPa]

    def surface_tension_star(self, temperature_star, quadrupole_star, bond_length_star):
        """Computes the reduced surface tension of the two-center
        Lennard-Jones model for a given set of model parameters over
        a specified range of temperatures.

        Parameters
        ----------
        temperature_star: numpy.ndarray
            The reduced temperatures to evaluate the reduced density at.
        quadrupole_star: float
            The reduced quadrupole parameter.
        bond_length_star: float
            The reduced bond-length parameter

        Returns
        -------
        numpy.ndarray
            The reduced surface tensions.
        """
        _B = self._B

        t_c_star = self.critical_temperature_star(quadrupole_star, bond_length_star)
        _a_correlation = self._a_correlation_function(quadrupole_star, bond_length_star)

        if any(temperature_star / t_c_star > 1.0):
            return np.empty(temperature_star.shape) * np.nan

        surface_tension_star = (
            _a_correlation * (1.0 - (temperature_star / t_c_star)) ** _B
        )
        return surface_tension_star

    def surface_tension(self, temperature, epsilon, sigma, bond_length, quadrupole):
        """Computes the surface tension of the two-center Lennard-Jones model
        for a given set of model parameters over a specified range of
        temperatures.

        Parameters
        ----------
        temperature: numpy.ndarray
            The temperatures to evaluate the density at in units of K.
        epsilon: float
            The epsilon parameter in units of K.
        sigma: float
            The sigma parameter in units of nm.
        bond_length: float
            The bond-length parameter in units of nm.
        quadrupole: float
            The quadrupole parameter in units of Debye * nm.

        Returns
        -------
        numpy.ndarray
            The evaluated surface tensions in units of J / m^2
        """

        # Note that epsilon is defined as epsilon/kB
        temperature_star = temperature / epsilon

        quadrupole_star_sqr = (quadrupole * self._reduced_D_to_sqrt_J_m3) ** 2 / (
            epsilon * self._reduced_boltzmann * sigma ** 5
        )

        bond_length_star = bond_length / sigma

        surface_tension_star = self.surface_tension_star(
            temperature_star, quadrupole_star_sqr, bond_length_star
        )
        surface_tension = (
            surface_tension_star
            * epsilon
            / sigma ** 2
            * self._reduced_boltzmann
            * 1.0e-5
        )
        return surface_tension  # [J/m2]

    def evaluate(self, property_type, parameters, temperatures):

        if property_type == NISTDataType.LiquidDensity:
            return self.liquid_density(temperatures, *parameters)
        elif property_type == NISTDataType.SaturationPressure:
            return self.saturation_pressure(temperatures, *parameters)
        elif property_type == NISTDataType.SurfaceTension:
            return self.surface_tension(temperatures, *parameters)

        raise NotImplementedError()

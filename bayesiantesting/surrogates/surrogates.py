import abc
import os

from autograd import numpy as np
import yaml
from pkg_resources import resource_filename

from bayesiantesting.datasets.nist import NISTDataType

# Conversion constants
k_B = 1.38065e-23  # [J/K]
N_A = 6.02214e23  # [1/mol]
m3_to_nm3 = 1e27
m2_to_nm2 = 1e18
gm_to_kg = 1.0 / 1000
J_to_kJ = 1.0 / 1000
J_per_m3_to_kPA = 1.0 / 1000
D_to_sqrtJm3 = 3.1623e-25


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

        b = self.critical_temperature_star_parameters

        t_c_star = (
            1 * b[0] +
            quadrupole_star_sqr ** 2 * b[1] +
            quadrupole_star_sqr ** 3 * b[2] +
            1.0 / (0.1 + bond_length_star ** 2) * b[3] +
            1.0 / (0.1 + bond_length_star ** 5) * b[4] +
            quadrupole_star_sqr ** 2 / (0.1 + bond_length_star ** 2) * b[5] +
            quadrupole_star_sqr ** 2 / (0.1 + bond_length_star ** 5) * b[6] +
            quadrupole_star_sqr ** 3 / (0.1 + bond_length_star ** 2) * b[7] +
            quadrupole_star_sqr ** 3 / (0.1 + bond_length_star ** 5) * b[8]
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
        quadrupole = quadrupole * D_to_sqrtJm3  # [(J*m3)^(1/2) nm]
        quadrupole_sqr = quadrupole ** 2 * m3_to_nm3  # [J*nm5]

        # Note that epsilon is defined as epsilon / kB
        quadrupole_star_sqr = quadrupole_sqr / (epsilon * k_B * sigma ** 5)
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

        b = self.density_star_parameters

        rho_c_star = (
            1 * b[0] +
            quadrupole_star ** 2 * b[1] +
            quadrupole_star ** 3 * b[2] +
            bond_length_star ** 2 / (0.11 + bond_length_star ** 2) * b[3] +
            bond_length_star ** 5 / (0.11 + bond_length_star ** 5) * b[4] +
            bond_length_star ** 2 * quadrupole_star ** 2 / (0.11 + bond_length_star ** 2) * b[5] +
            bond_length_star ** 5 * quadrupole_star ** 2 / (0.11 + bond_length_star ** 5) * b[6] +
            bond_length_star ** 2 * quadrupole_star ** 3 / (0.11 + bond_length_star ** 2) * b[7] +
            bond_length_star ** 5 * quadrupole_star ** 3 / (0.11 + bond_length_star ** 5) * b[8]
        )

        return rho_c_star

    @staticmethod
    def _correlation_function_1(quadrupole_star, bond_length_star, b):

        result = (
            1 * b[0] +
            quadrupole_star ** 2 * b[1] +
            quadrupole_star ** 3 * b[2] +
            bond_length_star ** 3 / (bond_length_star + 0.4) ** 3 * b[3] +
            bond_length_star ** 4 / (bond_length_star + 0.4) ** 5 * b[4] +
            quadrupole_star ** 2 * bond_length_star ** 2 / (bond_length_star + 0.4) * b[5] +
            quadrupole_star ** 2 * bond_length_star ** 3 / (bond_length_star + 0.4) ** 7 * b[6] +
            quadrupole_star ** 3 * bond_length_star ** 2 / (bond_length_star + 0.4) * b[7] +
            quadrupole_star ** 3 * bond_length_star ** 3 / (bond_length_star + 0.4) ** 7 * b[8]
        )

        return result

    @staticmethod
    def _correlation_function_2(quadrupole_star, bond_length_star, b):

        result = (
            1 * b[0] +
            quadrupole_star ** 2 * b[1] +
            quadrupole_star ** 3 * b[2] +
            bond_length_star ** 2 * b[3] +
            bond_length_star ** 3 * b[4] +
            quadrupole_star ** 2 * bond_length_star ** 2 * b[5] +
            quadrupole_star ** 2 * bond_length_star ** 3 * b[6] +
            quadrupole_star ** 3 * bond_length_star ** 2 * b[7]
        )

        return result

    @staticmethod
    def _correlation_function_3(quadrupole_star, bond_length_star, b):

        result = (
            1 * b[0] +
            quadrupole_star ** 2 * b[1] +
            quadrupole_star ** 3 * b[2] +
            bond_length_star * b[3] +
            bond_length_star ** 4 * b[4] +
            quadrupole_star ** 2 * bond_length_star * b[5] +
            quadrupole_star ** 2 * bond_length_star ** 4 * b[6] +
            quadrupole_star ** 3 * bond_length_star ** 4 * b[7]
        )

        return result

    def density_star(self, temperature_star, quadrupole_star, bond_length_star, phase):
        """Computes the reduced critical temperature of the two-center
        Lennard-Jones model in the specified phase for a given set of model parameters over
        a specified range of temperatures.

        Parameters
        ----------
        temperature_star: numpy.ndarray
            The reduced temperatures to evaluate the reduced density at.
        quadrupole_star: float
            The reduced quadrupole parameter.
        bond_length_star: float
            The reduced bond-length parameter
        phase: str
            The phase to compute the density of. This must be one of
            `'liquid'` or `'vapor'`.

        Returns
        -------
        numpy.ndarray
            The reduced density.
        """

        _b_C1, _b_C2_L, _b_C3_L, _b_C2_v, _b_C3_v = (
            self._b_C1,
            self._b_C2_L,
            self._b_C3_L,
            self._b_C2_v,
            self._b_C3_v,
        )

        critical_temperature_star = self.critical_temperature_star(
            quadrupole_star, bond_length_star
        )
        rho_c_star = self.critical_density_star(quadrupole_star, bond_length_star)

        tau = critical_temperature_star - temperature_star

        if all(tau > 0):

            coefficient_1 = self._correlation_function_1(quadrupole_star, bond_length_star, _b_C1)

            if phase == "liquid":

                coefficient_2 = self._correlation_function_2(quadrupole_star, bond_length_star, _b_C2_L)
                coefficient_3 = self._correlation_function_3(quadrupole_star, bond_length_star, _b_C3_L)

            elif phase == "vapor":

                coefficient_1 = -coefficient_1
                coefficient_2 = self._correlation_function_2(quadrupole_star, bond_length_star, _b_C2_v)
                coefficient_3 = self._correlation_function_3(quadrupole_star, bond_length_star, _b_C3_v)

            else:

                raise NotImplementedError()

            x_0 = 1.0 * rho_c_star
            x_1 = tau ** (1.0 / 3.0) * coefficient_1
            x_2 = tau * coefficient_2
            x_3 = tau ** (3.0 / 2.0) * coefficient_3

            rho_star = x_0 + x_1 + x_2 + x_3

        else:
            rho_star = np.zeros(len(tau))

        return rho_star

    def _density(self, temperature, epsilon, sigma, bond_length, quadrupole, phase):
        """Computes the density of the two-center Lennard-Jones model
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
        phase: str
            The phase to compute the density of. This must be one of
            `'liquid'` or `'vapor'`.

        Returns
        -------
        numpy.ndarray
            The evaluated densities in units of kg / m3.
        """

        molecular_weight = self.molecular_weight.magnitude

        # Note that epsilon is defined as epsilon/kB
        temperature_star = temperature / epsilon

        quadrupole = quadrupole * D_to_sqrtJm3  # [(J*m3)^(1/2) nm]
        quadrupole_sqr = quadrupole ** 2 * m3_to_nm3  # [J*nm5]
        quadrupole_star_sqr = quadrupole_sqr / (epsilon * k_B * sigma ** 5)

        bond_length_star = bond_length / sigma

        rho_star = self.density_star(
            temperature_star, quadrupole_star_sqr, bond_length_star, phase
        )
        rho = rho_star * molecular_weight / sigma ** 3 / N_A * m3_to_nm3 * gm_to_kg
        return rho  # [kg/m3]

    def liquid_density(self, temperature, epsilon, sigma, bond_length, quadrupole):
        """Computes the liquid-phase density of the two-center Lennard-Jones model
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
            The evaluated densities in units of kg / m3.
        """

        return self._density(
            temperature, epsilon, sigma, bond_length, quadrupole, "liquid"
        )  # [kg/m3]

    def vapor_density(self, temperature, epsilon, sigma, bond_length, quadrupole):
        """Computes the vapor-phase density of the two-center Lennard-Jones model
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
            The evaluated densities in units of kg / m3.
        """

        return self._density(
            temperature, epsilon, sigma, bond_length, quadrupole, "vapor"
        )  # [kg/m3]

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

        c1 = (
            1.0 * _b_c1[0] +
            quadrupole_star ** 2 * _b_c1[1] +
            quadrupole_star ** 3 * _b_c1[2] +
            bond_length_star ** 2 / (bond_length_star ** 2 + 0.75) * _b_c1[3] +
            bond_length_star ** 3 / (bond_length_star ** 3 + 0.75) * _b_c1[4] +
            bond_length_star ** 2 * quadrupole_star ** 2 / (bond_length_star ** 2 + 0.75) * _b_c1[5] +
            bond_length_star ** 3 * quadrupole_star ** 2 / (bond_length_star ** 3 + 0.75) * _b_c1[6] +
            bond_length_star ** 2 * quadrupole_star ** 3 / (bond_length_star ** 2 + 0.75) * _b_c1[7] +
            bond_length_star ** 3 * quadrupole_star ** 3 / (bond_length_star ** 3 + 0.75) * _b_c1[8]
        )
        c2 = (
            1.0 * _b_c2[0] +
            quadrupole_star ** 2 * _b_c2[1] +
            quadrupole_star ** 3 * _b_c2[2] +
            bond_length_star ** 2 / (bond_length_star + 0.75) ** 2 * _b_c2[3] +
            bond_length_star ** 3 / (bond_length_star + 0.75) ** 3 * _b_c2[4] +
            bond_length_star ** 2 * quadrupole_star ** 2 / (bond_length_star + 0.75) ** 2 * _b_c2[5] +
            bond_length_star ** 3 * quadrupole_star ** 2 / (bond_length_star + 0.75) ** 3 * _b_c2[6] +
            bond_length_star ** 2 * quadrupole_star ** 3 / (bond_length_star + 0.75) ** 2 * _b_c2[7] +
            bond_length_star ** 3 * quadrupole_star ** 3 / (bond_length_star + 0.75) ** 3 * _b_c2[8]
        )
        c3 = (
            quadrupole_star ** 2 * _b_c3[0] +
            quadrupole_star ** 5 * _b_c3[1] +
            bond_length_star ** 0.5
        )

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

        temperature_star = (
            temperature / epsilon
        )  # note that epsilon is defined as epsilon/kB

        quadrupole = quadrupole * D_to_sqrtJm3  # [(J*m3)^(1/2) nm]
        quadrupole_sqr = quadrupole ** 2 * m3_to_nm3  # [J*nm5]

        # Note that epsilon is defined as epsilon/kB
        quadrupole_star_sqr = quadrupole_sqr / (epsilon * k_B * sigma ** 5)

        bond_length_star = bond_length / sigma

        saturation_pressure_star = self.saturation_pressure_star(
            temperature_star, quadrupole_star_sqr, bond_length_star
        )
        saturation_pressure = (
            saturation_pressure_star
            * epsilon
            / sigma ** 3
            * k_B
            * m3_to_nm3
            * J_per_m3_to_kPA
        )
        return saturation_pressure  # [kPa]

    def _a_correlation_function(self, quadrupole_star, bond_length_star):

        c_a, c_b, c_c, c_d, c_e = self._A_a, self._A_b, self._A_c, self._A_d, self._A_e

        a = 1.0 * c_a
        b = (
            quadrupole_star * c_b[0] +
            quadrupole_star ** 2.0 * c_b[1] +
            quadrupole_star ** 3.0 * c_b[2]
        )
        c = 1.0 / (bond_length_star ** 2.0 + 0.1) * c_c[0]
        d = (
            quadrupole_star ** 2.0 * bond_length_star ** 2.0 * c_d[0] +
            quadrupole_star ** 2.0 * bond_length_star ** 3.0 * c_d[1]
        )
        e = (
            quadrupole_star ** 2 / (bond_length_star ** 2.0 + 0.1) * c_e[0] +
            quadrupole_star ** 2.0 / (bond_length_star ** 5.0 + 0.1) * c_e[1]
        )

        return a + b + c + d + e

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

        critical_temperature_star = self.critical_temperature_star(
            quadrupole_star, bond_length_star
        )

        _a_correlation = self._a_correlation_function(quadrupole_star, bond_length_star)
        surface_tension_star = (
            _a_correlation
            * (1.0 - (temperature_star / critical_temperature_star)) ** _B
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

        quadrupole = quadrupole * D_to_sqrtJm3  # [(J*m3)^(1/2) nm]
        quadrupole_sqr = quadrupole ** 2 * m3_to_nm3  # [J*nm5]
        quadrupole_star_sqr = quadrupole_sqr / (epsilon * k_B * sigma ** 5)

        bond_length_star = bond_length / sigma

        surface_tension_star = self.surface_tension_star(
            temperature_star, quadrupole_star_sqr, bond_length_star
        )
        surface_tension = surface_tension_star * epsilon / sigma ** 2 * k_B * m2_to_nm2
        return surface_tension  # [J/m2]

    def evaluate(self, property_type, parameters, temperatures):

        if property_type == NISTDataType.LiquidDensity:
            return self.liquid_density(temperatures, *parameters)
        elif property_type == NISTDataType.SaturationPressure:
            return self.saturation_pressure(temperatures, *parameters)
        elif property_type == NISTDataType.SurfaceTension:
            return self.surface_tension(temperatures, *parameters)

        raise NotImplementedError()

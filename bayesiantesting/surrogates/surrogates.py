import os

import numpy as np
import yaml

# Conversion constants
from pkg_resources import resource_filename

k_B = 1.38065e-23  # [J/K]
N_A = 6.02214e23  # [1/mol]
m3_to_nm3 = 1e27
m2_to_nm2 = 1e18
gm_to_kg = 1.0 / 1000
J_to_kJ = 1.0 / 1000
J_per_m3_to_kPA = 1.0 / 1000
D_to_sqrtJm3 = 3.1623e-25


class TwoCenterLennardJones:
    """A surrogate model for the two-center Lennard-Jones model, which can
    be rapidly evaluated from the models critical density and temperature,
    liquid and vapor density, saturation pressure and surface tension.
    """

    def __init__(self, molecular_weight, file_path=None):
        """Constructs a new `TwoCenterLennardJones` object.

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

    def critical_temperature_star(self, quadrupole_star, bond_length_star):
        """Computes the reduced critical temperature of the two-center
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
            The reduced critical temperature.
        """

        b = self.critical_temperature_star_parameters

        x = np.array(
            [
                1,
                quadrupole_star ** 2,
                quadrupole_star ** 3,
                1.0 / (0.1 + bond_length_star ** 2),
                1.0 / (0.1 + bond_length_star ** 5),
                quadrupole_star ** 2 / (0.1 + bond_length_star ** 2),
                quadrupole_star ** 2 / (0.1 + bond_length_star ** 5),
                quadrupole_star ** 3 / (0.1 + bond_length_star ** 2),
                quadrupole_star ** 3 / (0.1 + bond_length_star ** 5),
            ]
        )

        t_c_star = x * b
        t_c_star = t_c_star.sum()

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

        x = np.array(
            [
                1,
                quadrupole_star ** 2,
                quadrupole_star ** 3,
                bond_length_star ** 2 / (0.11 + bond_length_star ** 2),
                bond_length_star ** 5 / (0.11 + bond_length_star ** 5),
                bond_length_star ** 2
                * quadrupole_star ** 2
                / (0.11 + bond_length_star ** 2),
                bond_length_star ** 5
                * quadrupole_star ** 2
                / (0.11 + bond_length_star ** 5),
                bond_length_star ** 2
                * quadrupole_star ** 3
                / (0.11 + bond_length_star ** 2),
                bond_length_star ** 5
                * quadrupole_star ** 3
                / (0.11 + bond_length_star ** 5),
            ]
        )

        rho_c_star = x * b
        rho_c_star = rho_c_star.sum()
        return rho_c_star

    @staticmethod
    def _correlation_function_1(quadrupole_star, bond_length_star, b):

        result = np.array(
            [
                1,
                quadrupole_star ** 2,
                quadrupole_star ** 3,
                bond_length_star ** 3 / (bond_length_star + 0.4) ** 3,
                bond_length_star ** 4 / (bond_length_star + 0.4) ** 5,
                quadrupole_star ** 2 * bond_length_star ** 2 / (bond_length_star + 0.4),
                quadrupole_star ** 2
                * bond_length_star ** 3
                / (bond_length_star + 0.4) ** 7,
                quadrupole_star ** 3 * bond_length_star ** 2 / (bond_length_star + 0.4),
                quadrupole_star ** 3
                * bond_length_star ** 3
                / (bond_length_star + 0.4) ** 7,
            ]
        )
        result = result * b
        result = result.sum()

        return result

    @staticmethod
    def _correlation_function_2(quadrupole_star, bond_length_star, b):

        result = np.array(
            [
                1,
                quadrupole_star ** 2,
                quadrupole_star ** 3,
                bond_length_star ** 2,
                bond_length_star ** 3,
                quadrupole_star ** 2 * bond_length_star ** 2,
                quadrupole_star ** 2 * bond_length_star ** 3,
                quadrupole_star ** 3 * bond_length_star ** 2,
            ]
        )
        result = result * b
        result = result.sum()
        return result

    @staticmethod
    def _correlation_function_3(quadrupole_star, bond_length_star, b):

        result = np.array(
            [
                1,
                quadrupole_star ** 2,
                quadrupole_star ** 3,
                bond_length_star,
                bond_length_star ** 4,
                quadrupole_star ** 2 * bond_length_star,
                quadrupole_star ** 2 * bond_length_star ** 4,
                quadrupole_star ** 3 * bond_length_star ** 4,
            ]
        )
        result = result * b
        result = result.sum()

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

        tau = critical_temperature_star - temperature_star  # Tc* - T*

        if all(tau > 0):

            x = np.ones([len(tau), 4])  # First column is supposed to be all ones

            x[:, 1] = tau ** (1.0 / 3)
            x[:, 2] = tau
            x[:, 3] = tau ** (3.0 / 2)

            coefficient_1 = self._correlation_function_1(quadrupole_star, bond_length_star, _b_C1)

            if phase == "liquid":

                coefficient_2 = self._correlation_function_2(
                    quadrupole_star, bond_length_star, _b_C2_L
                )
                coefficient_3 = self._correlation_function_3(
                    quadrupole_star, bond_length_star, _b_C3_L
                )
                b = np.array([rho_c_star, coefficient_1, coefficient_2, coefficient_3])

            elif phase == "vapor":

                coefficient_2 = self._correlation_function_2(
                    quadrupole_star, bond_length_star, _b_C2_v
                )
                coefficient_3 = self._correlation_function_3(
                    quadrupole_star, bond_length_star, _b_C3_v
                )
                b = np.array([rho_c_star, -coefficient_1, coefficient_2, coefficient_3])

            else:
                return 0

            # rho_star = b[0]+b[1]*tau**(1./3)+b[2]*tau+b[3]*tau**(3./2) #The brute force approach
            rho_star = x * b
            rho_star = rho_star.sum(
                axis=1
            )  # To add up the rows (that pertain to a specific temperature_star)

        else:
            rho_star = np.zeros([len(tau)])

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

        molecular_weight = self.molecular_weight

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

        x_c1 = [
            1.0,
            quadrupole_star ** 2,
            quadrupole_star ** 3,
            bond_length_star ** 2 / (bond_length_star ** 2 + 0.75),
            bond_length_star ** 3 / (bond_length_star ** 3 + 0.75),
            bond_length_star ** 2
            * quadrupole_star ** 2
            / (bond_length_star ** 2 + 0.75),
            bond_length_star ** 3
            * quadrupole_star ** 2
            / (bond_length_star ** 3 + 0.75),
            bond_length_star ** 2
            * quadrupole_star ** 3
            / (bond_length_star ** 2 + 0.75),
            bond_length_star ** 3
            * quadrupole_star ** 3
            / (bond_length_star ** 3 + 0.75),
        ]
        x_c2 = [
            1.0,
            quadrupole_star ** 2,
            quadrupole_star ** 3,
            bond_length_star ** 2 / (bond_length_star + 0.75) ** 2,
            bond_length_star ** 3 / (bond_length_star + 0.75) ** 3,
            bond_length_star ** 2
            * quadrupole_star ** 2
            / (bond_length_star + 0.75) ** 2,
            bond_length_star ** 3
            * quadrupole_star ** 2
            / (bond_length_star + 0.75) ** 3,
            bond_length_star ** 2
            * quadrupole_star ** 3
            / (bond_length_star + 0.75) ** 2,
            bond_length_star ** 3
            * quadrupole_star ** 3
            / (bond_length_star + 0.75) ** 3,
        ]
        x_c3 = [quadrupole_star ** 2, quadrupole_star ** 5, bond_length_star ** 0.5]

        c1 = (x_c1 * _b_c1).sum()
        c2 = (x_c2 * _b_c2).sum()
        c3 = (x_c3 * _b_c3).sum()

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

        a, b, c, d, e = self._A_a, self._A_b, self._A_c, self._A_d, self._A_e

        x_a = np.array([1])
        x_b = np.array(
            [quadrupole_star, quadrupole_star ** 2.0, quadrupole_star ** 3.0]
        )
        x_c = np.array([1.0 / (bond_length_star ** 2.0 + 0.1)])
        x_d = np.array(
            [
                quadrupole_star ** 2.0 * bond_length_star ** 2.0,
                quadrupole_star ** 2.0 * bond_length_star ** 3.0,
            ]
        )
        x_e = np.array(
            [
                quadrupole_star ** 2 / (bond_length_star ** 2.0 + 0.1),
                quadrupole_star ** 2.0 / (bond_length_star ** 5.0 + 0.1),
            ]
        )

        _a_correlation = (x_a * a).sum()
        _a_correlation += (x_b * b).sum()
        _a_correlation += (x_c * c).sum()
        _a_correlation += (x_d * d).sum()
        _a_correlation += (x_e * e).sum()

        return _a_correlation

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

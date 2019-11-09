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
    """A two-center Lennard-Jones model.
    """

    def __init__(self, M_w, file_path=None):
        """Constructs a new `TwoCenterLennardJones` object.

        Parameters
        ----------
        M_w
        file_path: str, optional
            The path to the model parameters. If unset, the built in
            `DCLJQ.yaml` parameters will be used.
        """

        self.M_w = M_w

        if file_path is None:

            file_path = resource_filename(
                "bayesiantesting", os.path.join("data", "models", "DCLJQ.yaml")
            )

        with open(file_path) as file:

            parameters = yaml.load(file, Loader=yaml.SafeLoader)[
                "correlation_parameters"
            ]

        self.T_c_star_params = np.array(parameters["Stoll"]["T_c_star_params"])
        self.rho_c_star_params = np.array(parameters["Stoll"]["rho_c_star_params"])

        self.b_C1 = np.array(parameters["Stoll"]["rho_L_star_params"]["C1_params"])
        self.b_C2_L = np.array(parameters["Stoll"]["rho_L_star_params"]["C2_params"])
        self.b_C3_L = np.array(parameters["Stoll"]["rho_L_star_params"]["C3_params"])
        self.b_C2_v = np.array(parameters["Stoll"]["rho_v_star_params"]["C2_params"])
        self.b_C3_v = np.array(parameters["Stoll"]["rho_v_star_params"]["C3_params"])

        self.b_c1 = np.array(parameters["Stoll"]["P_v_star_params"]["c1_params"])
        self.b_c2 = np.array(parameters["Stoll"]["P_v_star_params"]["c2_params"])
        self.b_c3 = np.array(parameters["Stoll"]["P_v_star_params"]["c3_params"])

        self.A_a = np.array(parameters["Werth"]["A_star_params"]["a_params"])
        self.A_b = np.array(parameters["Werth"]["A_star_params"]["b_params"])
        self.A_c = np.array(parameters["Werth"]["A_star_params"]["c_params"])
        self.A_d = np.array(parameters["Werth"]["A_star_params"]["d_params"])
        self.A_e = np.array(parameters["Werth"]["A_star_params"]["e_params"])

        self.B = np.array(parameters["Werth"]["A_star_params"]["B_params"])

    def T_c_star_hat(self, q, l):

        b = self.T_c_star_params
        x = np.array(
            [
                1,
                q ** 2,
                q ** 3,
                1.0 / (0.1 + l ** 2),
                1.0 / (0.1 + l ** 5),
                q ** 2 / (0.1 + l ** 2),
                q ** 2 / (0.1 + l ** 5),
                q ** 3 / (0.1 + l ** 2),
                q ** 3 / (0.1 + l ** 5),
            ]
        )

        t_c_star = x * b
        t_c_star = t_c_star.sum()

        return t_c_star

    def rho_c_star_hat(self, q, l):
        b = self.rho_c_star_params
        x = np.array(
            [
                1,
                q ** 2,
                q ** 3,
                l ** 2 / (0.11 + l ** 2),
                l ** 5 / (0.11 + l ** 5),
                l ** 2 * q ** 2 / (0.11 + l ** 2),
                l ** 5 * q ** 2 / (0.11 + l ** 5),
                l ** 2 * q ** 3 / (0.11 + l ** 2),
                l ** 5 * q ** 3 / (0.11 + l ** 5),
            ]
        )
        rho_c_star = x * b
        rho_c_star = rho_c_star.sum()
        return rho_c_star

    def C1_hat(self, q, l, b):
        x_C1 = np.array(
            [
                1,
                q ** 2,
                q ** 3,
                l ** 3 / (l + 0.4) ** 3,
                l ** 4 / (l + 0.4) ** 5,
                q ** 2 * l ** 2 / (l + 0.4),
                q ** 2 * l ** 3 / (l + 0.4) ** 7,
                q ** 3 * l ** 2 / (l + 0.4),
                q ** 3 * l ** 3 / (l + 0.4) ** 7,
            ]
        )
        C1 = x_C1 * b
        C1 = C1.sum()
        return C1

    def C2_hat(self, q, l, b):
        x_C2 = np.array(
            [
                1,
                q ** 2,
                q ** 3,
                l ** 2,
                l ** 3,
                q ** 2 * l ** 2,
                q ** 2 * l ** 3,
                q ** 3 * l ** 2,
            ]
        )
        C2 = x_C2 * b
        C2 = C2.sum()
        return C2

    def C3_hat(self, q, l, b):
        x_C3 = np.array(
            [1, q ** 2, q ** 3, l, l ** 4, q ** 2 * l, q ** 2 * l ** 4, q ** 3 * l ** 4]
        )
        C3 = x_C3 * b
        C3 = C3.sum()
        return C3

    def rho_star_hat_2CLJQ(self, T_star, q, l, phase):
        b_C1, b_C2_L, b_C3_L, b_C2_v, b_C3_v = (
            self.b_C1,
            self.b_C2_L,
            self.b_C3_L,
            self.b_C2_v,
            self.b_C3_v,
        )
        T_c_star = self.T_c_star_hat(q, l)
        rho_c_star = self.rho_c_star_hat(q, l)
        tau = T_c_star - T_star  # T_c_star - T_star
        if all(tau > 0):
            x = np.ones([len(tau), 4])  # First column is supposed to be all ones
            x[:, 1] = tau ** (1.0 / 3)
            x[:, 2] = tau
            x[:, 3] = tau ** (3.0 / 2)
            C1 = self.C1_hat(q, l, b_C1)
            if phase == "liquid":
                C2 = self.C2_hat(q, l, b_C2_L)
                C3 = self.C3_hat(q, l, b_C3_L)
                b = np.array([rho_c_star, C1, C2, C3])
            elif phase == "vapor":
                C2 = self.C2_hat(q, l, b_C2_v)
                C3 = self.C3_hat(q, l, b_C3_v)
                b = np.array([rho_c_star, -C1, C2, C3])
            else:
                return 0
            # rho_star = b[0]+b[1]*tau**(1./3)+b[2]*tau+b[3]*tau**(3./2) #The brute force approach
            rho_star = x * b
            rho_star = rho_star.sum(
                axis=1
            )  # To add up the rows (that pertain to a specific T_star)
        else:
            rho_star = np.zeros([len(tau)])
        return rho_star

    def rho_hat_2CLJQ(self, Temp, eps, sig, Lbond, Qpole, phase):
        """
        inputs:
            Temp: temperature [K]
            eps: epsilon/kb [K]
            sig: sigma [nm]
            Lbond: bond-length [nm]
            Qpole: quadrupole [Debye * nm]
            phase: liquid or vapor
        outputs:
            rho: density [kg/m3]
        """

        M_w = self.M_w
        T_star = Temp / eps  # note that eps is defined as eps/kB
        Qpole = Qpole * D_to_sqrtJm3  # [(J*m3)^(1/2) nm]
        Q2pole = Qpole ** 2 * m3_to_nm3  # [J*nm5]
        Q2_star = Q2pole / (eps * k_B * sig ** 5)  # note that eps is defined as eps/kB
        L_star = Lbond / sig
        rho_star = self.rho_star_hat_2CLJQ(T_star, Q2_star, L_star, phase)
        rho = rho_star * M_w / sig ** 3 / N_A * m3_to_nm3 * gm_to_kg  # [kg/m3]
        return rho

    def rhol_hat_2CLJQ(self, Temp, eps, sig, Lbond, Qpole):
        rhol = self.rho_hat_2CLJQ(Temp, eps, sig, Lbond, Qpole, "liquid")
        return rhol  # [kg/m3]

    def rhov_hat_2CLJQ(self, Temp, eps, sig, Lbond, Qpole):
        rhov = self.rho_hat_2CLJQ(Temp, eps, sig, Lbond, Qpole, "vapor")
        return rhov  # [kg/m3]

    def Psat_star_hat_2CLJQ(self, T_star, q, l):
        b_c1, b_c2, b_c3 = self.b_c1, self.b_c2, self.b_c3
        x_c1 = [
            1.0,
            q ** 2,
            q ** 3,
            l ** 2 / (l ** 2 + 0.75),
            l ** 3 / (l ** 3 + 0.75),
            l ** 2 * q ** 2 / (l ** 2 + 0.75),
            l ** 3 * q ** 2 / (l ** 3 + 0.75),
            l ** 2 * q ** 3 / (l ** 2 + 0.75),
            l ** 3 * q ** 3 / (l ** 3 + 0.75),
        ]
        x_c2 = [
            1.0,
            q ** 2,
            q ** 3,
            l ** 2 / (l + 0.75) ** 2,
            l ** 3 / (l + 0.75) ** 3,
            l ** 2 * q ** 2 / (l + 0.75) ** 2,
            l ** 3 * q ** 2 / (l + 0.75) ** 3,
            l ** 2 * q ** 3 / (l + 0.75) ** 2,
            l ** 3 * q ** 3 / (l + 0.75) ** 3,
        ]
        x_c3 = [q ** 2, q ** 5, l ** 0.5]
        c1 = (x_c1 * b_c1).sum()
        c2 = (x_c2 * b_c2).sum()
        c3 = (x_c3 * b_c3).sum()
        Psat_star = np.exp(c1 + c2 / T_star + c3 / (T_star ** 4))
        return Psat_star

    def Psat_hat_2CLJQ(self, Temp, eps, sig, Lbond, Qpole):
        """
        inputs:
            Temp: temperature [K]
            eps: epsilon/kb [K]
            sig: sigma [nm]
            Lbond: bond-length [nm]
            Qpole: quadrupole [Debye * nm]
        outputs:
            Psat: vapor pressure [kPa]
        """

        T_star = Temp / eps  # note that eps is defined as eps/kB
        Qpole = Qpole * D_to_sqrtJm3  # [(J*m3)^(1/2) nm]
        Q2pole = Qpole ** 2 * m3_to_nm3  # [J*nm5]
        Q2_star = Q2pole / (eps * k_B * sig ** 5)  # note that eps is defined as eps/kB
        L_star = Lbond / sig
        Psat_star = self.Psat_star_hat_2CLJQ(T_star, Q2_star, L_star)
        Psat = (
            Psat_star * eps / sig ** 3 * k_B * m3_to_nm3 * J_per_m3_to_kPA
        )  # [kPa] #note that eps is defined as eps/kB
        return Psat

    def LJ_model(self, r, eps, sig):
        r_star = r / sig
        U = 4 * eps * (r_star ** (-12) - r_star ** (-6))
        return U

    def Astar_hat(self, q, l):
        a, b, c, d, e = self.A_a, self.A_b, self.A_c, self.A_d, self.A_e
        x_a = np.array([1])
        x_b = np.array([q, q ** 2.0, q ** 3.0])
        x_c = np.array([1.0 / (l ** 2.0 + 0.1)])
        x_d = np.array([q ** 2.0 * l ** 2.0, q ** 2.0 * l ** 3.0])
        x_e = np.array([q ** 2 / (l ** 2.0 + 0.1), q ** 2.0 / (l ** 5.0 + 0.1)])
        Astar = (x_a * a).sum()
        Astar += (x_b * b).sum()
        Astar += (x_c * c).sum()
        Astar += (x_d * d).sum()
        Astar += (x_e * e).sum()
        return Astar

    def ST_star_hat_2CLJQ(self, T_star, q, l):
        B = self.B
        T_c_star = self.T_c_star_hat(q, l)
        Astar = self.Astar_hat(q, l)
        ST_star = Astar * (1.0 - (T_star / T_c_star)) ** B
        return ST_star

    def ST_hat_2CLJQ(self, Temp, eps, sig, Lbond, Qpole):
        """
        inputs:
            Temp: temperature [K]
            eps: epsilon/kb [K]
            sig: sigma [nm]
            Lbond: bond-length [nm]
            Qpole: quadrupole [Debye * nm]
        outputs:
            ST: surface tnesion [J/m2]
        """

        T_star = Temp / eps  # note that eps is defined as eps/kB
        Qpole = Qpole * D_to_sqrtJm3  # [(J*m3)^(1/2) nm]
        Q2pole = Qpole ** 2 * m3_to_nm3  # [J*nm5]
        Q2_star = Q2pole / (eps * k_B * sig ** 5)  # note that eps is defined as eps/kB
        L_star = Lbond / sig
        ST_star = self.ST_star_hat_2CLJQ(T_star, Q2_star, L_star)
        ST = (
            ST_star * eps / sig ** 2 * k_B * m2_to_nm2
        )  # [J/m2] #note that eps is defined as eps/kB
        return ST

    def T_c_hat_2CLJQ(self, eps, sig, Lbond, Qpole):
        """
        inputs:
            eps: epsilon/kb [K]
            sig: sigma [nm]
            Lbond: bond-length [nm]
            Qpole: quadrupole [Debye * nm]
        outputs:
            T_c: critical temperature [K]
        """

        Qpole = Qpole * D_to_sqrtJm3  # [(J*m3)^(1/2) nm]
        Q2pole = Qpole ** 2 * m3_to_nm3  # [J*nm5]
        Q2_star = Q2pole / (eps * k_B * sig ** 5)  # note that eps is defined as eps/kB
        L_star = Lbond / sig
        T_c_star = self.T_c_star_hat(Q2_star, L_star)
        T_c = T_c_star * eps
        return T_c

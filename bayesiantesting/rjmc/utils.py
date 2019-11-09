#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:08:09 2019

@author: owenmadin
"""

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymbar
import scipy as sp
import yaml
from scipy.optimize import curve_fit
from scipy.stats import expon
from statsmodels.stats.proportion import multinomial_proportions_confint


from bayesiantesting.utils import get_data_filename


def parse_data_ffs(compound):

    fname = get_data_filename(os.path.join("forcefields", f"{compound}.yaml"))

    with open(fname) as yfile:
        yfile = yaml.load(yfile, Loader=yaml.SafeLoader)

    ff_params = []
    params = ["eps_lit", "sig_lit", "Lbond_lit", "Q_lit"]

    for name in params:
        ff_params.append(yfile["force_field_params"][name])

    ff_params_ref = np.transpose(np.asarray(ff_params))
    ff_params_ref[:, 1:] = ff_params_ref[:, 1:] / 10

    Tc_lit = np.loadtxt(
        get_data_filename(os.path.join("trc_data", compound, "Tc.txt")), skiprows=1
    )
    M_w = np.loadtxt(
        get_data_filename(os.path.join("trc_data", compound, "Mw.txt")), skiprows=1
    )

    df = pd.read_csv(get_data_filename("nist_bond_lengths.txt"), delimiter="\t")
    df = df[df.Compound == compound]
    NIST_bondlength = np.asarray(df)

    data = ["rhoL", "Pv", "SurfTens"]
    data_dict = {}

    for name in data:

        df = pd.read_csv(
            get_data_filename(os.path.join("trc_data", compound, f"{name}.txt")),
            sep="\t",
        )
        df = df.dropna()
        data_dict[name] = df

    return ff_params_ref, Tc_lit, M_w, data_dict, NIST_bondlength[0][1] / 10


def filter_thermo_data(thermo_data, T_min, T_max, n_points):

    for name in thermo_data:
        df = thermo_data[name]

        df = df[df.values[:, 0] > T_min]
        df = df[df.values[:, 0] < T_max]
        if int(np.floor(df.shape[0] / (n_points - 1))) == 0:
            slicer = 1
        else:
            slicer = int(np.floor(df.shape[0] / (n_points - 1)))
        # print(slicer)
        df = df[::slicer]
        thermo_data[name] = df

    return thermo_data


def calculate_uncertainties(thermo_data, T_c):

    u_dict = {}

    for name in thermo_data:

        # Extract data from our data arrays
        data = np.asarray(thermo_data[name])
        T = data[:, 0]
        values = data[:, 1]
        u_exp = data[:, 2]

        pu_corr = uncertainty_models(T, T_c, name)
        u_corr = pu_corr * values

        u_tot = np.sqrt(u_corr ** 2 + u_exp ** 2)
        u_dict[name] = u_tot

    return u_dict


def uncertainty_models(T, T_c, thermo_property):

    Tr = T / T_c
    u = np.zeros(np.size(Tr))

    # Linear models for uncertainties in the 2CLJQ correlation we are using,
    # determined from Messerly analysis of figure from Stobener, Stoll, Werth

    # Starts at 0.3% for low values and ramps up to 1% for large values
    if thermo_property == "rhoL":
        for i in range(np.size(Tr)):
            if Tr[i] < 0.9:
                u[i] = 0.3
            elif 0.9 <= Tr[i] <= 0.95:
                u[i] = 0.3 + (1 - 0.3) * (Tr[i] - 0.9) / (0.95 - 0.9)
            else:
                u[i] = 1.0

    # Starts at 20% for low values and ramps down to 2% for large values
    if thermo_property == "Pv":
        for i in range(np.size(Tr)):
            if Tr[i] <= 0.55:
                u[i] = 20
            elif 0.55 <= Tr[i] <= 0.7:
                u[i] = 20 + (2 - 20) * (Tr[i] - 0.55) / (0.7 - 0.55)
            else:
                u[i] = 2.0

    # Starts at 4% for low values and ramps up to 12% for higher values
    if thermo_property == "SurfTens":
        for i in range(np.size(Tr)):
            if Tr[i] <= 0.75:
                u[i] = 4
            elif 0.75 <= Tr[i] <= 0.95:
                u[i] = 4 + (12 - 4) * (Tr[i] - 0.75) / (0.95 - 0.75)
            else:
                u[i] = 12.0
    u /= 100
    return u


def computePercentDeviations(
    compound_2CLJ,
    temp_values_rhol,
    temp_values_psat,
    temp_values_surftens,
    parameter_values,
    rhol_data,
    psat_data,
    surftens_data,
    T_c_data,
    rhol_hat_models,
    Psat_hat_models,
    SurfTens_hat_models,
    T_c_hat_models,
):

    rhol_model = rhol_hat_models(compound_2CLJ, temp_values_rhol, *parameter_values)
    psat_model = Psat_hat_models(compound_2CLJ, temp_values_psat, *parameter_values)
    if len(surftens_data) != 0:
        surftens_model = SurfTens_hat_models(
            compound_2CLJ, temp_values_surftens, *parameter_values
        )
        surftens_deviation_vector = (
            (surftens_data - surftens_model) / surftens_data
        ) ** 2
        surftens_mean_relative_deviation = (
            np.sqrt(
                np.sum(surftens_deviation_vector) / np.size(surftens_deviation_vector)
            )
            * 100
        )
    else:
        surftens_mean_relative_deviation = 0
    T_c_model = T_c_hat_models(compound_2CLJ, *parameter_values)

    rhol_deviation_vector = ((rhol_data - rhol_model) / rhol_data) ** 2
    psat_deviation_vector = ((psat_data - psat_model) / psat_data) ** 2

    T_c_relative_deviation = (T_c_data - T_c_model) * 100 / T_c_data

    rhol_mean_relative_deviation = (
        np.sqrt(np.sum(rhol_deviation_vector) / np.size(rhol_deviation_vector)) * 100
    )
    psat_mean_relative_deviation = (
        np.sqrt(np.sum(psat_deviation_vector) / np.size(psat_deviation_vector)) * 100
    )

    return (
        rhol_mean_relative_deviation,
        psat_mean_relative_deviation,
        surftens_mean_relative_deviation,
        T_c_relative_deviation,
    )


def rhol_hat_models(compound_2CLJ, Temp, model, eps, sig, L, Q):
    """
    L_nm=L/10
    sig_nm=sig/10
    Q_nm=Q/10
    """
    if model == 0:  # Two center AUA LJ

        rhol_hat = compound_2CLJ.rhol_hat_2CLJQ(Temp, eps, sig, L, 0)

    elif model == 1:  # Two center AUA LJ+Q

        rhol_hat = compound_2CLJ.rhol_hat_2CLJQ(Temp, eps, sig, L, Q)

    elif model == 2:  # 2CLJ model

        rhol_hat = compound_2CLJ.rhol_hat_2CLJQ(Temp, eps, sig, L, 0)

    else:
        raise NotImplementedError()

    return rhol_hat  # [kg/m3]


def Psat_hat_models(compound_2CLJ, Temp, model, eps, sig, L, Q):
    """
    L_nm=L/10
    sig_nm=sig/10
    Q_nm=Q/10
    """
    if model == 0:  # Two center AUA LJ

        Psat_hat = compound_2CLJ.Psat_hat_2CLJQ(Temp, eps, sig, L, 0)

    elif model == 1:  # Two center AUA LJ+Q

        Psat_hat = compound_2CLJ.Psat_hat_2CLJQ(Temp, eps, sig, L, Q)

    elif model == 2:  # 2CLJ model

        Psat_hat = compound_2CLJ.Psat_hat_2CLJQ(Temp, eps, sig, L, 0)

    else:
        raise NotImplementedError()

    return Psat_hat  # [kPa]


def SurfTens_hat_models(compound_2CLJ, Temp, model, eps, sig, L, Q):
    """
    L_nm=L/10
    sig_nm=sig/10
    Q_nm=Q/10
    """
    if model == 0:

        SurfTens_hat = compound_2CLJ.ST_hat_2CLJQ(Temp, eps, sig, L, 0)

    elif model == 1:

        SurfTens_hat = compound_2CLJ.ST_hat_2CLJQ(Temp, eps, sig, L, Q)

    elif model == 2:

        # Model 2 is the same as model 0, but the L value will be previously specified (not varying)
        SurfTens_hat = compound_2CLJ.ST_hat_2CLJQ(Temp, eps, sig, L, 0)

    else:
        raise NotImplementedError()

    return SurfTens_hat


def T_c_hat_models(compound_2CLJ, model, eps, sig, L, Q):
    """
    L_nm=L/10
    sig_nm=sig/10
    Q_nm=Q/10
    """
    if model == 0:

        T_c_hat = compound_2CLJ.T_c_hat_2CLJQ(eps, sig, L, 0)

    elif model == 1:

        T_c_hat = compound_2CLJ.T_c_hat_2CLJQ(eps, sig, L, Q)

    elif model == 2:

        T_c_hat = compound_2CLJ.T_c_hat_2CLJQ(eps, sig, L, 0)

    else:
        raise NotImplementedError()

    return T_c_hat


def create_param_triangle_plot_4D(
    trace, tracename, lit_values, properties, compound, n_iter, file_loc=None
):  # ,sig_prior,eps_prior,L_prior,Q_prior):
    if np.shape(trace) != (0,):
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle(
            "Parameter Marginal Distributions, "
            + compound
            + ", "
            + properties
            + ", "
            + str(n_iter)
            + " steps",
            fontsize=20,
        )

        axs[0, 0].hist(
            trace[:, 1], bins=50, color="m", density=True, label="RJMC Sampling"
        )
        axs[1, 1].hist(trace[:, 2], bins=50, color="m", density=True)
        axs[2, 2].hist(trace[:, 3], bins=50, color="m", density=True)
        axs[3, 3].hist(trace[:, 4], bins=50, color="m", density=True)

        """
        sig_prior=np.multiply(sig_prior,10)
        L_prior=np.multiply(L_prior,10)
        Q_prior=np.multiply(Q_prior,10)

        sig_range=np.linspace(0.5*min(trace[:,1]),2*max(trace[:,1]),num=100)
        eps_range=np.linspace(0.5*min(trace[:,2]),2*max(trace[:,2]),num=100)
        L_range=np.linspace(0.5*min(trace[:,3]),2*max(trace[:,3]),num=100)

        logitpdf=distributions.logistic.pdf
        """
        # axs[0,0].plot(sig_range,1000000000*logitpdf(sig_range,*sig_prior))
        # axs[1,1].plot(eps_range,1000000*logitpdf(eps_range,*eps_prior))
        # axs[2,2].plot(L_range,10*logitpdf(L_range,*L_prior))

        """
        axs[0,0].axvline(x=eps_prior[0],color='r',linestyle='--',label='Uniform Prior')
        axs[0,0].axvline(x=eps_prior[1],color='r',linestyle='--')
        axs[1,1].axvline(x=sig_prior[0],color='r',linestyle='--')
        axs[1,1].axvline(x=sig_prior[1],color='r',linestyle='--')
        axs[2,2].axvline(x=L_prior[0],color='r',linestyle='--')
        axs[2,2].axvline(x=L_prior[1],color='r',linestyle='--')
        """
        # axs[3,3].axvline(x=Q_prior[0],color='r',linestyle='--')
        # axs[3,3].axvline(x=Q_prior[1],color='r',linestyle='--')

        axs[0, 1].hist2d(
            trace[:, 2], trace[:, 1], bins=100, cmap="cool", label="RJMC Sampling"
        )
        axs[0, 2].hist2d(trace[:, 3], trace[:, 1], bins=100, cmap="cool")
        axs[0, 3].hist2d(trace[:, 4], trace[:, 1], bins=100, cmap="cool")
        axs[1, 2].hist2d(trace[:, 3], trace[:, 2], bins=100, cmap="cool")
        axs[1, 3].hist2d(trace[:, 4], trace[:, 2], bins=100, cmap="cool")
        axs[2, 3].hist2d(trace[:, 4], trace[:, 3], bins=100, cmap="cool")

        axs[0, 1].scatter(
            lit_values[::4, 1],
            lit_values[::4, 0],
            color="0.25",
            marker="o",
            alpha=0.5,
            facecolors="none",
            label="Pareto Values",
        )
        axs[0, 2].scatter(
            lit_values[::4, 2],
            lit_values[::4, 0],
            color="0.25",
            marker="o",
            alpha=0.5,
            facecolors="none",
        )
        axs[0, 3].scatter(
            lit_values[::4, 3],
            lit_values[::4, 0],
            color="0.25",
            marker="o",
            alpha=0.5,
            facecolors="none",
        )
        axs[1, 2].scatter(
            lit_values[::4, 2],
            lit_values[::4, 1],
            color="0.25",
            marker="o",
            alpha=0.5,
            facecolors="none",
        )
        axs[1, 3].scatter(
            lit_values[::4, 3],
            lit_values[::4, 1],
            color="0.25",
            marker="o",
            alpha=0.5,
            facecolors="none",
        )
        axs[2, 3].scatter(
            lit_values[::4, 3],
            lit_values[::4, 2],
            color="0.25",
            marker="o",
            alpha=0.5,
            facecolors="none",
        )

        # axs[0,1].set_ylim([min(lit_values[:,0]),max(lit_values[:,0])])

        fig.delaxes(axs[1, 0])
        fig.delaxes(axs[2, 0])
        fig.delaxes(axs[3, 0])
        fig.delaxes(axs[2, 1])
        fig.delaxes(axs[3, 1])
        fig.delaxes(axs[3, 2])
        """
        axs[0,0].axes.get_yaxis().set_visible(False)
        axs[1,1].axes.get_yaxis().set_visible(False)
        axs[2,2].axes.get_yaxis().set_visible(False)
        axs[3,3].axes.get_yaxis().set_visible(False)
        """
        axs[0, 1].axes.get_yaxis().set_visible(False)
        axs[0, 2].axes.get_yaxis().set_visible(False)
        axs[1, 2].axes.get_yaxis().set_visible(False)
        axs[1, 3].axes.get_xaxis().set_visible(False)
        axs[2, 3].axes.get_xaxis().set_visible(False)

        axs[0, 0].xaxis.tick_top()
        axs[0, 1].xaxis.tick_top()
        axs[0, 2].xaxis.tick_top()
        axs[0, 3].xaxis.tick_top()
        axs[0, 3].yaxis.tick_right()
        axs[1, 3].yaxis.tick_right()
        axs[2, 3].yaxis.tick_right()

        axs[0, 0].set_yticklabels([])
        axs[1, 1].set_yticklabels([])
        axs[2, 2].set_yticklabels([])
        axs[3, 3].set_yticklabels([])

        axs[0, 0].set_ylabel(r"$\epsilon$ (K)", fontsize=14)
        axs[1, 1].set_ylabel(r"$\sigma$ ($\AA$)", fontsize=14)
        axs[2, 2].set_ylabel(r"L ($\AA$)", fontsize=14)
        axs[3, 3].set_ylabel(r"Q (D$\AA$)", fontsize=14)

        axs[0, 0].set_xlabel(r"$\epsilon$ (K)", fontsize=14)
        axs[0, 1].set_xlabel(r"$\sigma$ ($\AA$)", fontsize=14)
        axs[0, 2].set_xlabel(r"L ($\AA$)", fontsize=14)
        axs[0, 3].set_xlabel(r"Q (D$\AA$)", fontsize=14)

        axs[0, 0].xaxis.set_label_position("top")
        axs[0, 1].xaxis.set_label_position("top")
        axs[0, 2].xaxis.set_label_position("top")
        axs[0, 3].xaxis.set_label_position("top")

        handles, labels = axs[0, 1].get_legend_handles_labels()
        handles0, labels0 = axs[0, 0].get_legend_handles_labels()
        # plt.figlegend((label0,label1),('Literature','RJMC Sampling'))
        fig.legend(handles, labels, loc=[0.1, 0.4])
        plt.savefig(file_loc + tracename + ".png")
        plt.close()
        # plt.show()

    return


def create_percent_dev_triangle_plot(
    trace, tracename, lit_values, properties, compound, n_iter, file_loc=None
):
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(
        "Percent Deviation Marginal Distributions, "
        + compound
        + ", "
        + properties
        + ", "
        + str(n_iter)
        + " steps"
    )
    axs[0, 0].hist(trace[:, 0], bins=50, color="m", density=True)
    axs[1, 1].hist(trace[:, 1], bins=50, color="m", density=True)
    axs[2, 2].hist(trace[:, 2], bins=50, color="m", density=True)
    axs[3, 3].hist(trace[:, 3], bins=50, color="m", density=True)

    axs[0, 1].hist2d(trace[:, 1], trace[:, 0], bins=100, cmap="cool")
    axs[0, 2].hist2d(trace[:, 2], trace[:, 0], bins=100, cmap="cool")
    axs[0, 3].hist2d(trace[:, 3], trace[:, 0], bins=100, cmap="cool")
    axs[1, 2].hist2d(trace[:, 2], trace[:, 1], bins=100, cmap="cool")
    axs[1, 3].hist2d(trace[:, 3], trace[:, 1], bins=100, cmap="cool")
    axs[2, 3].hist2d(trace[:, 3], trace[:, 2], bins=100, cmap="cool")

    axs[0, 1].scatter(
        lit_values[::4, 1],
        lit_values[::4, 0],
        color="0.25",
        marker="o",
        alpha=0.5,
        facecolors="none",
        label="Stobener Pareto Values",
    )
    axs[0, 2].scatter(
        lit_values[::4, 2],
        lit_values[::4, 0],
        color="0.25",
        marker="o",
        alpha=0.5,
        facecolors="none",
    )
    axs[0, 3].scatter(
        lit_values[::4, 3],
        lit_values[::4, 0],
        color="0.25",
        marker="o",
        alpha=0.5,
        facecolors="none",
    )
    axs[1, 2].scatter(
        lit_values[::4, 2],
        lit_values[::4, 1],
        color="0.25",
        marker="o",
        alpha=0.5,
        facecolors="none",
    )
    axs[1, 3].scatter(
        lit_values[::4, 3],
        lit_values[::4, 1],
        color="0.25",
        marker="o",
        alpha=0.5,
        facecolors="none",
    )
    axs[2, 3].scatter(
        lit_values[::4, 3],
        lit_values[::4, 2],
        color="0.25",
        marker="o",
        alpha=0.5,
        facecolors="none",
    )

    # axs[0,1].set_xlim([min(lit_values[::4,1]),max(lit_values[::4,1])])
    # axs[0,1].set_ylim([min(lit_values[::4,0]),max(lit_values[::4,0])])

    fig.delaxes(axs[1, 0])
    fig.delaxes(axs[2, 0])
    fig.delaxes(axs[3, 0])
    fig.delaxes(axs[2, 1])
    fig.delaxes(axs[3, 1])
    fig.delaxes(axs[3, 2])

    axs[0, 1].axes.get_yaxis().set_visible(False)
    axs[0, 2].axes.get_yaxis().set_visible(False)
    axs[1, 2].axes.get_yaxis().set_visible(False)
    axs[1, 3].axes.get_xaxis().set_visible(False)
    axs[2, 3].axes.get_xaxis().set_visible(False)

    axs[0, 0].xaxis.tick_top()
    axs[0, 1].xaxis.tick_top()
    axs[0, 2].xaxis.tick_top()
    axs[0, 3].xaxis.tick_top()
    axs[0, 3].yaxis.tick_right()
    axs[1, 3].yaxis.tick_right()
    axs[2, 3].yaxis.tick_right()

    axs[0, 0].set_yticklabels([])
    axs[1, 1].set_yticklabels([])
    axs[2, 2].set_yticklabels([])
    axs[3, 3].set_yticklabels([])

    axs[0, 0].set(ylabel=r"% Deviation, $\rho_l$")
    axs[1, 1].set(ylabel=r"% Deviation, $P_{sat}$")
    axs[2, 2].set(ylabel=r"% Deviation, $\gamma$")
    axs[3, 3].set(ylabel=r"% Deviation, $T_c$")

    axs[0, 0].set(xlabel=r"% Deviation, $\rho_l$")
    axs[0, 1].set(xlabel=r"% Deviation, $P_{sat}$")
    axs[0, 2].set(xlabel=r"% Deviation, $\gamma$")
    axs[0, 3].set(xlabel=r"% Deviation, $T_c$")

    axs[0, 0].xaxis.set_label_position("top")
    axs[0, 1].xaxis.set_label_position("top")
    axs[0, 2].xaxis.set_label_position("top")
    axs[0, 3].xaxis.set_label_position("top")

    handles, labels = axs[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc=[0.05, 0.3])
    plt.savefig(file_loc + tracename + ".png")
    plt.close()
    # plt.show()


def plot_bar_chart(prob, properties, compound, n_iter, n_models, file_loc=None):
    x = np.arange(n_models)
    prob = prob[-1:] + prob[:-1]
    print(prob)
    prob_copy = copy.deepcopy(prob)
    basis = min(i for i in prob if i > 0)
    # while basis==0:
    # prob_copy=np.delete(prob_copy,np.argmin(prob))
    # if len(prob_copy)==0:
    #    basis=1
    # else:
    #    basis=min(prob_copy)
    value = prob / basis
    if np.size(prob) == 2:
        color = ["red", "blue"]
        label = "AUA,AUA+Q"
    elif np.size(prob) == 3:
        color = ["red", "blue", "orange"]
        label = ("UA", "AUA", "AUA+Q")
    plt.bar(x, value, color=["red", "blue", "orange"])
    plt.xticks(x, ("UA", "AUA", "AUA+Q"), fontsize=14)
    plt.title(
        "Model Bayes Factor, "
        + compound
        + ", "
        + properties
        + ", "
        + str(n_iter)
        + " steps",
        fontsize=14,
    )
    plt.ylabel("Bayes Factor", fontsize=14)

    plt.savefig(file_loc + "/bar_chart.png")
    plt.close()
    # plt.show()
    return


def import_literature_values(criteria, compound):
    df = pd.read_csv(
        get_data_filename(
            os.path.join("literature", f"Pareto_Hasse_{criteria}_criteria.txt")
        ),
        delimiter=" ",
        skiprows=2,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    )

    df = df[df.Substance == compound]
    df1 = df.iloc[:, 1:5]
    df2 = df.iloc[:, 5:9]
    df1 = df1[["epsilon", "sigma", "L", "Q"]]

    return np.asarray(df1), np.asarray(df2)
    # return df1,df2


def compute_multinomial_confidence_intervals(trace):
    indices = pymbar.timeseries.subsampleCorrelatedData(trace[::10, 0])

    confint_trace_USE = trace[indices]

    trace_model_0 = []
    trace_model_1 = []
    trace_model_2 = []
    for i in range(np.size(confint_trace_USE, 0)):
        if confint_trace_USE[i, 0] == 0:
            trace_model_0.append(confint_trace_USE[i])
            # log_trace_0.append(logp_trace[i])
        elif confint_trace_USE[i, 0] == 1:
            trace_model_1.append(confint_trace_USE[i])
            # log_trace_1.append(logp_trace[i])
        elif confint_trace_USE[i, 0] == 2:
            trace_model_2.append(confint_trace_USE[i])
            # log_trace_2.append(logp_trace[i])

    trace_model_0 = np.asarray(trace_model_0)
    trace_model_1 = np.asarray(trace_model_1)
    trace_model_2 = np.asarray(trace_model_2)

    counts = np.asarray([len(trace_model_0), len(trace_model_1), len(trace_model_2)])

    prob_conf = multinomial_proportions_confint(counts)

    return prob_conf


def unbias_simulation(biasing_factor, probabilities):
    unbias_prob = probabilities * np.exp(-biasing_factor)
    unbias_prob_normalized = unbias_prob / sum(unbias_prob)

    return unbias_prob_normalized


def fit_exponential_sp(trace, plot=False):
    loc, scale = expon.fit(trace[:, 4])
    if plot == True:
        xmax = max(trace[:, 4])
        xmin = min(trace[:, 4])
        xdata = np.linspace(xmin, xmax, num=500)
        plt.plot(xdata, expon.pdf(xdata, loc, scale))
        plt.hist(trace[:, 4], bins=50, density=True)
    return loc, scale


def fit_gamma(trace, bins=25):
    # DONT USE
    y, x = np.histogram(trace, bins=bins, density=True)
    x_adjust = []
    for i in range(len(x) - 1):
        x_adjust.append((x[i] + x[i + 1]) / 2)

    def func(x, a, b):
        return (
            (1 / (sp.special.gamma(a) * (b ** a))) * np.power(x, a - 1) * np.exp(-x / b)
        )

    popt, pcov = curve_fit(func, x_adjust, y, bounds=(0, [500, 400]))
    plt.plot(x_adjust, func(x_adjust, *popt))
    plt.plot(x_adjust, y)
    plt.show()
    return popt

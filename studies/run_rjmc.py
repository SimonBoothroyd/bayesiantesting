#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:42:37 2019

@author: owenmadin
"""

from datetime import date

import yaml

from bayesiantesting import models, rjmc


def parse_input_yaml(filepath):

    print("Loading simulation params from " + filepath + "...")

    with open(filepath) as file:
        simulation_params = yaml.load(file, Loader=yaml.SafeLoader)

    return simulation_params


def main():

    print("Parsing simulation params")
    simulation_params = parse_input_yaml("basic_run.yaml")

    print(simulation_params["priors"])

    prior = rjmc.RJMCPrior(simulation_params["priors"])
    prior.epsilon_prior()
    prior.sigma_prior()
    prior.L_prior()
    prior.Q_prior()

    rjmc_simulator = rjmc.RJMCSimulation(
        simulation_params["compound"],
        simulation_params["trange"],
        simulation_params["properties"],
        simulation_params["number_data_points"],
        simulation_params["steps"],
        simulation_params["swap_freq"],
        simulation_params["biasing_factor"],
        simulation_params["optimum_matching"],
    )

    rjmc_simulator.prepare_data()

    print("Simulation Attributes:", rjmc_simulator.get_attributes())

    compound_2CLJ = models.TwoCenterLennardJones(rjmc_simulator.M_w)
    rjmc_simulator.optimum_bounds = simulation_params["opt_bounds"]
    rjmc_simulator.gen_Tmatrix(prior, compound_2CLJ)
    # print(rjmc_simulator.opt_params_AUA)
    rjmc_simulator.set_initial_state(prior, compound_2CLJ)

    rjmc_simulator.RJMC_Outerloop(prior, compound_2CLJ)
    trace, logp_trace, percent_dev_trace, BAR_trace = rjmc_simulator.Report(
        USE_BAR=simulation_params["USE_BAR"]
    )

    rjmc_simulator.write_output(
        simulation_params["priors"],
        tag=simulation_params["label"],
        save_traj=simulation_params["save_traj"],
    )

    path = (
        "output/"
        + simulation_params["compound"]
        + "/"
        + simulation_params["properties"]
        + "/"
        + simulation_params["compound"]
        + "_"
        + simulation_params["properties"]
        + "_"
        + str(simulation_params["steps"])
        + "_"
        + simulation_params["label"]
        + "_"
        + str(date.today())
        + "/runfile.yaml"
    )

    with open(path, "w") as outfile:
        yaml.dump(simulation_params, outfile, default_flow_style=False)

    print("Finished!")


if __name__ == "__main__":
    main()

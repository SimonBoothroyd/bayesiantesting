"""
Code to perform MCMC simulations on simple toy models.
This code was originally authored by Owen Madin (github name ocmadin).
"""
import math

import numpy as np
import pymc3.distributions
from tqdm import tqdm


class MCMCSimulation:
    """ Builds an object that runs an MCMC simulation.
    """

    def __init__(
        self, model, warm_up_steps=100000, steps=100000, tune_frequency=5000,
    ):
        """Initializes the basic state of the simulator object.

        Parameters
        ----------
        model: Model
            The model whose posterior should be sampled.
        warm_up_steps: int
            The number of warm-up steps to take. During this time all
            data is discarded and the move proposals will be tuned.
        steps: int
            The number of steps which the simulation should run for.
        tune_frequency: int
            The frequency with which to tune the move proposals.
        """

        self.warm_up_steps = warm_up_steps
        self.steps = steps

        self.tune_frequency = tune_frequency

        self.optimum_bounds = "Normal"

        self.model = model

    def set_initial_state(self, initial_values=None):

        initial_log_p = math.nan

        if initial_values is not None:
            initial_values = np.copy(initial_values)

        else:

            counter = 0

            while math.isnan(initial_log_p) and counter < 1000:

                initial_values = self.model.sample_priors()
                initial_log_p = self.model.evaluate_log_posterior(initial_values)

                counter += 1

        print(f"Markov Chain initialized values:", initial_values)
        print("==============================")

        initial_log_p = self.model.evaluate_log_posterior(initial_values)

        print("Initial log posterior:", initial_log_p)
        print("==============================")

        if np.isnan(initial_log_p):

            raise ValueError(
                "The initial values could not be set without yielding "
                "a NaN log posterior"
            )

        # initial_percent_deviation = utils.computePercentDeviations(
        #     surrogate,
        #     self.thermo_data_rhoL[:, 0],
        #     self.thermo_data_Pv[:, 0],
        #     self.thermo_data_SurfTens[:, 0],
        #     initial_values,
        #     self.thermo_data_rhoL[:, 1],
        #     self.thermo_data_Pv[:, 1],
        #     self.thermo_data_SurfTens[:, 1],
        #     self.Tc_lit[0],
        #     utils.rhol_hat_models,
        #     utils.Psat_hat_models,
        #     utils.SurfTens_hat_models,
        #     utils.T_c_hat_models,
        # )

        return initial_values, initial_log_p

    def run(self, initial_values):

        trace = [initial_values]
        log_p_trace = [self.model.evaluate_log_posterior(initial_values)]
        # percent_dev_trace = [self.initial_percent_deviation]

        move_proposals = np.zeros((1, 1))
        move_acceptances = np.zeros((1, 1))

        proposal_scales = np.asarray(initial_values) / 100

        print("Initializing Simulation...")
        print("Tuning Proposals...")
        print("==============================")

        for i in tqdm(range(self.steps)):

            if not i % 50000:
                # print('Iteration ' + str(i)), print('Log Posterior:', self.logp_trace[i])
                pass

            current_params = trace[i].copy()
            current_log_prob = log_p_trace[i].copy()

            new_params, new_log_prob, acceptance = self._run_step(
                current_params, proposal_scales, current_log_prob
            )

            move_proposals[0, 0] += 1

            if acceptance == "True":
                move_acceptances[0, 0] += 1

            log_p_trace.append(new_log_prob)
            trace.append(new_params)

            # percent_dev_trace.append(
            #     utils.computePercentDeviations(
            #         surrogate,
            #         self.thermo_data_rhoL[:, 0],
            #         self.thermo_data_Pv[:, 0],
            #         self.thermo_data_SurfTens[:, 0],
            #         self.trace[i + 1],
            #         self.thermo_data_rhoL[:, 1],
            #         self.thermo_data_Pv[:, 1],
            #         self.thermo_data_SurfTens[:, 1],
            #         self.Tc_lit[0],
            #         utils.rhol_hat_models,
            #         utils.Psat_hat_models,
            #         utils.SurfTens_hat_models,
            #         utils.T_c_hat_models,
            #     )
            # )

            if (not (i + 1) % self.tune_frequency) and (i < self.warm_up_steps):

                proposal_scales = self._tune_proposals(
                    move_proposals, move_acceptances, proposal_scales
                )

            if i == self.warm_up_steps:

                move_proposals = np.zeros((1, 1))
                move_acceptances = np.zeros((1, 1))

        trace = np.asarray(trace)
        log_p_trace = np.asarray(log_p_trace)
        # percent_dev_trace = np.asarray(percent_dev_trace)

        print("Simulation Done!")
        print("==============================")

        return trace, log_p_trace

    def _run_step(self, current_params, proposal_scales, current_log_prob):

        proposed_params = current_params.copy()

        proposed_params, proposed_log_prob = self.parameter_proposal(
            proposed_params, proposal_scales
        )
        alpha = proposed_log_prob - current_log_prob

        acceptance = self._accept_reject(alpha)

        if acceptance is True:

            new_log_prob = proposed_log_prob
            new_params = proposed_params

        else:

            new_log_prob = current_log_prob
            new_params = current_params

        return new_params, new_log_prob, acceptance

    def parameter_proposal(self, proposed_params, proposal_scales):

        # Choose a random parameter to change
        parameter_index = int(
            np.ceil(np.random.random() * self.model.number_of_parameters)
        )

        # Sample the new parameters from a normal distribution.
        proposal_distribution = pymc3.distributions.Normal.dist(
            proposed_params[parameter_index], proposal_scales[parameter_index]
        )
        proposed_params[parameter_index] = proposal_distribution.random()

        proposed_log_prob = self.model.evaluate_log_posterior(proposed_params)

        return proposed_params, proposed_log_prob

    @staticmethod
    def _accept_reject(alpha):

        # Metropolis-Hastings accept/reject criteria
        random_number = pymc3.distributions.Uniform.dist(0.0, 1.0).random()
        return np.log(random_number) < alpha

    @staticmethod
    def _tune_proposals(move_proposals, move_acceptances, proposal_scales):

        # print(np.sum(self.move_proposals))
        acceptance_rate = np.sum(move_acceptances) / np.sum(move_proposals)

        # print(acceptance_rate)

        if acceptance_rate < 0.2:

            proposal_scales *= 0.9
            # print('Yes')

        elif acceptance_rate > 0.5:

            proposal_scales *= 1.1
            # print('No')

        return proposal_scales

    # def write_output(self, prior_dict, tag=None, save_traj=False):
    #
    #     # Ask if output exists
    #     if os.path.isdir("output") is False:
    #         os.mkdir("output")
    #     if os.path.isdir("output/" + self.compound) is False:
    #         os.mkdir("output/" + self.compound)
    #     if os.path.isdir("output/" + self.compound + "/" + self.properties) is False:
    #         os.mkdir("output/" + self.compound + "/" + self.properties)
    #
    #     path = (
    #         "output/"
    #         + self.compound
    #         + "/"
    #         + self.properties
    #         + "/"
    #         + self.compound
    #         + "_"
    #         + self.properties
    #         + "_"
    #         + str(self.steps)
    #         + "_"
    #         + tag
    #         + "_"
    #         + str(date.today())
    #     )
    #
    #     if os.path.isdir(path):
    #         print("Directory Exists, overwriting")
    #         rmtree(path)
    #         os.mkdir(path)
    #     else:
    #         os.mkdir(path)
    #
    #     os.mkdir(path + "/figures")
    #
    #     print("Creating figures...")
    #     print("==============================")
    #     plt.plot(self.logp_trace_tuned)
    #     plt.savefig(path + "/figures/logp_trace.png")
    #     plt.close()
    #
    #     plt.plot(self.trace_tuned[:, 0])
    #     plt.savefig(path + "/figures/model_trace.png")
    #     plt.close()
    #
    #     utils.create_param_triangle_plot_4D(
    #         self.trace_model_0,
    #         "triangle_plot_trace_model_0",
    #         self.lit_params,
    #         self.properties,
    #         self.compound,
    #         self.steps,
    #         file_loc=path + "/figures/",
    #     )
    #     utils.create_param_triangle_plot_4D(
    #         self.trace_model_1,
    #         "triangle_plot_trace_model_1",
    #         self.lit_params,
    #         self.properties,
    #         self.compound,
    #         self.steps,
    #         file_loc=path + "/figures/",
    #     )
    #     utils.create_param_triangle_plot_4D(
    #         self.trace_model_2,
    #         "triangle_plot_trace_model_2",
    #         self.lit_params,
    #         self.properties,
    #         self.compound,
    #         self.steps,
    #         file_loc=path + "/figures/",
    #     )
    #     utils.create_percent_dev_triangle_plot(
    #         self.percent_dev_trace_tuned,
    #         "triangle_plot_percent_dev_trace",
    #         self.lit_devs,
    #         self.properties,
    #         self.compound,
    #         self.steps,
    #         file_loc=path + "/figures/",
    #     )
    #     utils.plot_bar_chart(
    #         self.prob,
    #         self.properties,
    #         self.compound,
    #         self.steps,
    #         self.n_models,
    #         file_loc=path + "/figures/",
    #     )
    #
    #     print("Writing metadata...")
    #     print("==============================")
    #     self.write_datapoints(path)
    #
    #     self.write_metadata(path, prior_dict)
    #
    #     self.write_simulation_results(path)
    #
    #     if save_traj:
    #         print("Saving Trajectories")
    #         print("==============================")
    #         self.write_traces(path)
    #
    # def get_attributes(self):
    #     """Return attributes of RJMC system
    #     """
    #
    #     return {
    #         "compound": self.compound,
    #         "properties": self.properties,
    #         "T_range": self.T_range,
    #         "n_points": self.n_points,
    #         "steps": self.steps,
    #         "swap_freq": self.swap_freq,
    #         "biasing_factor": self.biasing_factor,
    #     }
    #
    #
    # def Report(self, plotting=False, USE_BAR=False):
    #     print("Proposed Moves:")
    #     print(np.sum(self.move_proposals))
    #     print(self.move_proposals)
    #     print("==============================")
    #     print("Successful Moves:")
    #     print(self.move_acceptances)
    #     print("==============================")
    #     prob_matrix = self.move_acceptances / self.move_proposals
    #     print("Ratio of successful moves")
    #     print(prob_matrix)
    #     print("==============================")
    #     transition_matrix = np.ones((3, 3))
    #     transition_matrix[0, 1] = (
    #         self.move_acceptances[0, 1] / np.sum(self.move_proposals, 1)[0]
    #     )
    #     transition_matrix[0, 2] = (
    #         self.move_acceptances[0, 2] / np.sum(self.move_proposals, 1)[0]
    #     )
    #     transition_matrix[1, 0] = (
    #         self.move_acceptances[1, 0] / np.sum(self.move_proposals, 1)[1]
    #     )
    #     transition_matrix[1, 2] = (
    #         self.move_acceptances[1, 2] / np.sum(self.move_proposals, 1)[1]
    #     )
    #     transition_matrix[2, 1] = (
    #         self.move_acceptances[2, 1] / np.sum(self.move_proposals, 1)[2]
    #     )
    #     transition_matrix[2, 0] = (
    #         self.move_acceptances[2, 0] / np.sum(self.move_proposals, 1)[2]
    #     )
    #     transition_matrix[0, 0] = 1 - transition_matrix[0, 1] - transition_matrix[0, 2]
    #     transition_matrix[1, 1] = 1 - transition_matrix[1, 0] - transition_matrix[1, 2]
    #     transition_matrix[2, 2] = 1 - transition_matrix[2, 0] - transition_matrix[2, 1]
    #     print("Transition Matrix:")
    #     print(transition_matrix)
    #     print("==============================")
    #     self.transition_matrix = transition_matrix
    #
    #     self.trace_tuned = self.trace[self.tune_for + 1 :]
    #     self.logp_trace_tuned = self.logp_trace[self.tune_for + 1 :]
    #     self.percent_dev_trace_tuned = self.percent_dev_trace[self.tune_for + 1 :]
    #
    #     self.lit_params, self.lit_devs = utils.import_literature_values(
    #         "two", self.compound
    #     )
    #     trace_equil = self.trace_tuned
    #     logp_trace_equil = self.logp_trace_tuned
    #     percent_dev_trace_equil = self.percent_dev_trace_tuned
    #     self.prob_conf = None
    #     try:
    #         self.prob_conf = utils.compute_multinomial_confidence_intervals(trace_equil)
    #     except pymbar.utils.ParameterError:
    #         print("Cannot compute confidence intervals due to only sampling one model")
    #
    #     # Converts the array with number of model parameters into an array with
    #     # the number of times there was 1 parameter or 2 parameters
    #     model_count = np.array(
    #         [
    #             len(trace_equil[trace_equil[:, 0] == 0]),
    #             len(trace_equil[trace_equil[:, 0] == 1]),
    #             len(trace_equil[trace_equil[:, 0] == 2]),
    #         ]
    #     )
    #
    #     prob_0 = 1.0 * model_count[0] / (len(trace_equil))
    #     print(
    #         "Percent that  model 0 is sampled: " + str(prob_0 * 100.0)
    #     )  # The percent that use 1 parameter model
    #
    #     prob_1 = 1.0 * model_count[1] / (len(trace_equil))
    #     print(
    #         "Percent that model 1 is sampled: " + str(prob_1 * 100.0)
    #     )  # The percent that use two center UA LJ
    #
    #     prob_2 = 1.0 * model_count[2] / (len(trace_equil))
    #     print(
    #         "Percent that model 2 is sampled: " + str(prob_2 * 100.0)
    #     )  # The percent that use two center UA LJ
    #     print("==============================")
    #     self.prob = [prob_0, prob_1, prob_2]
    #
    #     self.Exp_ratio = [prob_0 / prob_1, prob_0 / prob_2]
    #
    #     if self.prob_conf is not None:
    #         print("95% confidence intervals for probability", self.prob_conf)
    #
    #     self.unbiased_prob = utils.unbias_simulation(
    #         np.asarray(self.biasing_factor), np.asarray(self.prob)
    #     )
    #     print("Unbiased probabilities")
    #
    #     print("Experimental sampling ratio:", self.Exp_ratio)
    #     print("==============================")
    #
    #     print("Detailed Balance")
    #
    #     # These sets of numbers should be roughly equal to each other (If both
    #     # models are sampled).  If not, big problem
    #
    #     print(prob_0 * transition_matrix[0, 1])
    #     print(prob_1 * transition_matrix[1, 0])
    #
    #     print(prob_0 * transition_matrix[0, 2])
    #     print(prob_2 * transition_matrix[2, 0])
    #
    #     print(prob_1 * transition_matrix[1, 2])
    #     print(prob_2 * transition_matrix[2, 1])
    #     print("==============================")
    #     trace_model_0 = []
    #     trace_model_1 = []
    #     trace_model_2 = []
    #     """
    #     log_trace_0=[]
    #     log_trace_1=[]
    #     log_trace_2=[]
    #     """
    #     for i in range(np.size(trace_equil, 0)):
    #         if trace_equil[i, 0] == 0:
    #             trace_model_0.append(trace_equil[i])
    #             # log_trace_0.append(logp_trace[i])
    #         elif trace_equil[i, 0] == 1:
    #             trace_model_1.append(trace_equil[i])
    #             # log_trace_1.append(logp_trace[i])
    #         elif trace_equil[i, 0] == 2:
    #             trace_model_2.append(trace_equil[i])
    #             # log_trace_2.append(logp_trace[i])
    #
    #     self.trace_model_0 = np.asarray(trace_model_0)
    #     self.trace_model_1 = np.asarray(trace_model_1)
    #     self.trace_model_2 = np.asarray(trace_model_2)
    #
    #     self.BAR_trace = np.asarray(self.BAR_trace)
    #     if USE_BAR is True:
    #         self.BF_BAR = self.compute_BAR()
    #         print("BAR Bayes factor estimates")
    #         print(self.BF_BAR)
    #
    #     else:
    #         self.BF_BAR = None
    #
    #     if plotting:
    #
    #         utils.create_param_triangle_plot_4D(
    #             self.trace_model_0,
    #             "trace_model_0",
    #             self.lit_params,
    #             self.properties,
    #             self.compound,
    #             self.steps,
    #         )
    #         utils.create_param_triangle_plot_4D(
    #             self.trace_model_1,
    #             "trace_model_1",
    #             self.lit_params,
    #             self.properties,
    #             self.compound,
    #             self.steps,
    #         )
    #         utils.create_param_triangle_plot_4D(
    #             self.trace_model_2,
    #             "trace_model_2",
    #             self.lit_params,
    #             self.properties,
    #             self.compound,
    #             self.steps,
    #         )
    #
    #         utils.create_percent_dev_triangle_plot(
    #             percent_dev_trace_equil,
    #             "percent_dev_trace",
    #             self.lit_devs,
    #             self.prob,
    #             self.properties,
    #             self.compound,
    #             self.steps,
    #         )
    #
    #     return (
    #         self.trace_tuned,
    #         self.logp_trace_tuned,
    #         self.percent_dev_trace_tuned,
    #         self.BAR_trace,
    #     )
    #
    # def write_datapoints(self, path):
    #
    #     datapoints = {
    #         "Density Temperatures": self.thermo_data_rhoL[:, 0],
    #         "Density Values": self.thermo_data_rhoL[:, 1],
    #         "Density Measurement Uncertainties": self.thermo_data_rhoL[:, 2],
    #         "Saturation Pressure Temperatures": self.thermo_data_Pv[:, 0],
    #         "Saturation Pressure Values": self.thermo_data_Pv[:, 1],
    #         "Saturation Pressure Uncertainties": self.thermo_data_Pv[:, 2],
    #         "Surface Tension Temperatures": self.thermo_data_SurfTens[:, 0],
    #         "Surface Tension Values": self.thermo_data_SurfTens[:, 1],
    #         "Surface Tension Uncertainties": self.thermo_data_SurfTens[:, 2],
    #         "Literature Critical Temperature": self.Tc_lit[0],
    #     }
    #
    #     filename = path + "/datapoints.pkl"
    #
    #     with open(filename, "wb") as f:
    #         pickle.dump(datapoints, f)
    #
    # def write_metadata(self, path, prior_dict):
    #
    #     metadata = self.get_attributes()
    #     filename = path + "/metadata.pkl"
    #
    #     with open(filename, "wb") as f:
    #         pickle.dump(metadata, f)
    #
    # def write_simulation_results(self, path):
    #     results = {
    #         "Proposed Moves": self.move_proposals,
    #         "Tuning Frequency": self.tune_freq,
    #         "Tuning Length": self.tune_for,
    #         "Final Move SD": self.prop_sd,
    #         "Accepted Moves": self.move_acceptances,
    #         "Transition Matrix": self.transition_matrix,
    #         "Model Probabilities": self.prob,
    #         "Timestamp": str(datetime.today()),
    #         "Bayes Factors (Sampling Ratio)": self.Exp_ratio,
    #         "Model Probability confidence intervals": self.prob_conf,
    #         "Unbiased Probabilities": self.unbiased_prob,
    #     }
    #     if self.BF_BAR is not None:
    #         results["Bayes Factors (BAR)"] = self.BF_BAR
    #
    #     filename = path + "/results.pkl"
    #     with open(filename, "wb") as f:
    #         pickle.dump(results, f)
    #
    # def write_traces(self, path):
    #     if os.path.isdir(path + "/trace") == False:
    #         os.mkdir(path + "/trace")
    #     np.save(path + "/trace/trace.npy", self.trace_tuned)
    #     np.save(path + "/trace/logp_trace.npy", self.logp_trace_tuned)
    #     np.save(
    #         path + "/trace/percent_dev_trace_tuned.npy", self.percent_dev_trace_tuned
    #     )
    #     np.save(path + "/trace/BAR_trace.npy", self.BAR_trace)

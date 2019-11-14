"""
Code to perform MCMC simulations on simple toy models.
This code was originally authored by Owen Madin (github name ocmadin).
"""

import numpy as np
import torch
import torch.distributions
from bayesiantesting.models import Model, ModelCollection
from tqdm import tqdm


class MCMCSimulation:
    """ Builds an object that runs an MCMC simulation.
    """

    @property
    def model(self):
        """ModelCollection: The model being simulated.
        """
        return self._model

    def __init__(
        self,
        model,
        warm_up_steps=100000,
        steps=100000,
        tune_frequency=5000,
        discard_warm_up_data=True,
    ):
        """Initializes the basic state of the simulator object.

        Parameters
        ----------
        model: Model or ModelCollection
            The model whose posterior should be sampled.
        warm_up_steps: int
            The number of warm-up steps to take. During this time all
            move proposals will be tuned.
        steps: int
            The number of steps which the simulation should run for.
        tune_frequency: int
            The frequency with which to tune the move proposals.
        discard_warm_up_data: bool
            If true, all data generated during the warm-up period will
            be discarded.
        """

        self.warm_up_steps = warm_up_steps
        self.steps = steps

        self._tune_frequency = tune_frequency
        self._discard_warm_up_data = discard_warm_up_data

        self.optimum_bounds = "Normal"

        if isinstance(model, Model):
            # Convert any standalone models to model collections
            # for convenience.
            model = ModelCollection(model.name, (model,))

        self._model = model

        self._initial_values = None
        self._initial_model_index = None
        self._initial_log_p = None

    def _validate_parameter_shapes(self, initial_parameters, initial_model_index):

        if initial_model_index < 0 or initial_model_index >= self._model.n_models:

            raise ValueError(f'The model index was outside the allowed range '
                             f'[0, {self._model.n_models})')

        maximum_n_parameters = 0

        for model in self._model.models:
            maximum_n_parameters = max(maximum_n_parameters, model.n_total_parameters)

        if len(initial_parameters) != maximum_n_parameters:

            raise ValueError(f'The initial parameters vector is too small '
                             f'{len(initial_parameters)} to store the maximum '
                             f'number of the parameters from across all models '
                             f'({maximum_n_parameters}).')

    def run(self, initial_parameters, initial_model_index=0):

        # Make sure the parameters are the correct shape for the
        # specified model.
        self._validate_parameter_shapes(initial_parameters, initial_model_index)

        self._initial_values = initial_parameters
        self._initial_model_index = initial_model_index

        initial_model = self.model.models[initial_model_index]

        initial_log_p = initial_model.evaluate_log_posterior(self._initial_values)
        initial_deviations = initial_model.compute_percentage_deviations(self._initial_values)

        # Initialize the trace vectors
        total_steps = self.steps + self.warm_up_steps

        trace = np.zeros((total_steps + 1, len(self._initial_values) + 1))
        trace[0, 0] = self._initial_model_index
        trace[0, 1:] = self._initial_values

        log_p_trace = np.zeros(total_steps + 1)
        log_p_trace[0] = initial_log_p

        percent_deviation_trace = [initial_deviations]

        print(f"Markov Chain initialized values:", initial_parameters)
        print("==============================")

        print("Initial log posterior:", log_p_trace[-1])
        print("==============================")

        move_proposals = np.zeros((self._model.n_models, self._model.n_models))
        move_acceptances = np.zeros((self._model.n_models, self._model.n_models))

        proposal_scales = np.asarray(self._initial_values) / 100

        print("Initializing Simulation...")
        print("==============================")

        for i in tqdm(range(self.warm_up_steps + self.steps)):

            current_model_index = trace[i][0]
            current_parameters = trace[i][1:]

            current_log_prob = log_p_trace[i]
            current_percent_deviation = percent_deviation_trace[i]

            # Propose the new state.
            new_parameters, new_model_index, new_log_p, acceptance = self._run_step(
                current_parameters, current_model_index, proposal_scales, current_log_prob
            )
            new_percent_deviation = current_percent_deviation

            # Update the bookkeeping.
            move_proposals[current_model_index, new_model_index] += 1

            if acceptance:
                move_acceptances[current_model_index, new_model_index] += 1
                new_percent_deviation = self.model.models[new_model_index].compute_percentage_deviations(
                    new_parameters
                )

            trace[i + 1][0] = new_model_index
            trace[i + 1][1:] = new_parameters

            log_p_trace[i + 1] = new_log_p

            if i > self.warm_up_steps or not self._discard_warm_up_data:
                percent_deviation_trace.append(new_percent_deviation)

            if (not (i + 1) % self._tune_frequency) and (i < self.warm_up_steps):

                proposal_scales = self._tune_proposals(
                    move_proposals, move_acceptances, proposal_scales
                )

            if i == self.warm_up_steps:

                move_proposals = np.zeros((self._model.n_models, self._model.n_models))
                move_acceptances = np.zeros((self._model.n_models, self._model.n_models))

                if self._discard_warm_up_data:

                    # Discard any warm-up data.
                    trace = trace[i + 1:]
                    log_p_trace = log_p_trace[i + 1:]

        percent_deviation_trace_arrays = {
            label: np.zeros(len(percent_deviation_trace))
            for label in percent_deviation_trace[0]
        }

        for label in percent_deviation_trace_arrays:

            for index in range(len(percent_deviation_trace)):
                percent_deviation_trace_arrays[label][index] = percent_deviation_trace[
                    index
                ][label]

        print("Simulation Done!")
        print("==============================")

        print(f"Markov Chain final values:", trace[-1])
        print("==============================")

        return trace, log_p_trace, percent_deviation_trace_arrays

    def _run_step(self, current_params, current_model_index, proposal_scales, current_log_prob):

        proposed_params = current_params.copy()

        proposed_params, proposed_log_prob = self.parameter_proposal(
            proposed_params, proposal_scales
        )
        alpha = proposed_log_prob - current_log_prob

        acceptance = self._accept_reject(alpha)

        if acceptance:

            new_log_prob = proposed_log_prob
            new_params = proposed_params

        else:

            new_log_prob = current_log_prob
            new_params = current_params

        return new_params, current_model_index, new_log_prob, acceptance

    def parameter_proposal(self, proposed_params, current_model_index, proposal_scales):

        model = self.model

        # Choose a random parameter to change
        parameter_index = torch.randint(self.model.n_trainable_parameters, (1,))

        # Sample the new parameters from a normal distribution.
        proposed_params[parameter_index] = torch.distributions.Normal(
            proposed_params[parameter_index], proposal_scales[parameter_index]
        ).sample()

        if isinstance(self.model, Model):
            proposed_log_prob = self.model.evaluate_log_posterior(proposed_params)
        elif isinstance(self.model, ModelCollection):
            proposed_log_prob = self.model.evaluate_log_posterior(current_model_index, proposed_params)
        else:
            raise NotImplementedError()

        return proposed_params, proposed_log_prob

    @staticmethod
    def _accept_reject(alpha):

        # Metropolis-Hastings accept/reject criteria
        random_number = torch.rand((1,))
        return torch.log(random_number).item() < alpha

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
    #     run_identifier = f'{self.model.name}_{self.steps}_{tag}_{str(date.today())}'
    #
    #     figure_directory = os.path.join('output', run_identifier, 'figures')
    #     os.makedirs(figure_directory, exist_ok=True)
    #
    #     pyplot.plot(self.logp_trace_tuned)
    #     pyplot.savefig(os.path.join(figure_directory, "logp_trace.png"))
    #     pyplot.close()
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
    #     # print("Writing metadata...")
    #     # print("==============================")
    #     # self.write_datapoints(path)
    #     #
    #     # self.write_metadata(path, prior_dict)
    #     #
    #     # self.write_simulation_results(path)
    #     #
    #     # if save_traj:
    #     #     print("Saving Trajectories")
    #     #     print("==============================")
    #     #     self.write_traces(path)
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

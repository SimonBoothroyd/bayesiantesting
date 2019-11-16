"""
Code to perform RJMC simulations on simple toy models.
This code was originally authored by Owen Madin (github name ocmadin).
"""
import math

import numpy as np
import torch
from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.models.models import ModelCollection


class RJMCSimulation(MCMCSimulation):
    """ Builds an object that runs an RJMC simulation based
    on the parameters the user gives to it
    """

    def __init__(
        self,
        model_collection,
        warm_up_steps=100000,
        steps=100000,
        tune_frequency=5000,
        discard_warm_up_data=True,
        sampler=None,
        swap_frequency=0.3,
    ):
        """
        Parameters
        ----------
        swap_frequency: float
            The percentage of times the simulation tries to jump between models.
        """
        if not isinstance(model_collection, ModelCollection):
            raise ValueError("The model must be a `ModelCollection`.")

        if not model_collection.n_models > 1:
            raise ValueError(
                "The model collection must contain at least two "
                "sub-models to jump between."
            )

        if sampler is not None:
            raise ValueError('Samplers cannot currently be used with RJMC')

        super().__init__(
            model_collection, warm_up_steps, steps, tune_frequency, discard_warm_up_data, sampler
        )

        self._swap_frequency = swap_frequency

    def _run_step(
        self,
        current_parameters,
        current_model_index,
        proposal_scales,
        current_log_p,
        move_proposals,
        move_acceptances,
        adapt=False
    ):

        proposed_parameters = current_parameters.copy()
        proposed_model_index = int(current_model_index)

        random_move = torch.rand((1,)).item()

        if random_move <= self._swap_frequency:

            # Propose a cross-model move.
            (
                proposed_parameters,
                proposed_log_p,
                proposed_model_index,
                jacobian,
                transition_probability,
            ) = self.model_proposal(proposed_parameters, proposed_model_index)

            alpha = (
                (proposed_log_p - current_log_p)
                + np.log(jacobian)
                + np.log(transition_probability)
            )

        else:

            # Propose an in-model move.
            proposed_parameters, proposed_log_p = self.parameter_proposal(
                proposed_parameters, proposed_model_index, proposal_scales
            )
            alpha = proposed_log_p - current_log_p

        # Check for NaNs in the proposed state.
        if proposed_log_p == math.nan:

            alpha = -math.inf
            proposed_log_p = -math.inf

        # Apply the acceptance criteria.
        acceptance = self._accept_reject(alpha)

        move_proposals[current_model_index, proposed_model_index] += 1

        if acceptance:

            new_log_p = proposed_log_p
            new_params = proposed_parameters
            new_model_index = proposed_model_index

            move_acceptances[current_model_index, new_model_index] += 1

        else:

            new_log_p = current_log_p
            new_params = current_parameters
            new_model_index = current_model_index

        return new_params, new_model_index, new_log_p, acceptance

    def model_proposal(self, current_parameters, current_model_index):

        proposed_model_index = int(current_model_index)

        # Propose new model to jump to
        while proposed_model_index == current_model_index:
            proposed_model_index = torch.randint(
                self._model_collection.n_models, (1,)
            ).item()

        _, proposed_parameters, jacobian_array = self._model_collection.map_parameters(
            current_parameters, current_model_index, proposed_model_index
        )

        proposed_log_p = self._model_collection.evaluate_log_posterior(
            proposed_model_index, proposed_parameters
        )

        jacobian = np.prod(jacobian_array)
        transition_probability = self._model_collection.transition_probabilities(
            current_model_index, proposed_model_index
        )

        # Return values of jacobian in order to properly calculate accept/reject
        return (
            proposed_parameters,
            proposed_log_p,
            proposed_model_index,
            jacobian,
            transition_probability,
        )

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

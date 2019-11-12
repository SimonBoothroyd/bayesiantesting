"""
Code to perform RJMC simulations on simple toy models.
This code was originally authored by Owen Madin (github name ocmadin).
"""
# import copy
# import math
#
# import numpy as np
# from bayesiantesting.kernels import MCMCSimulation
# from bayesiantesting.models.models import ModelCollection


# class RJMCSimulation(MCMCSimulation):
#     """ Builds an object that runs an RJMC simulation based
#     on the parameters the user gives to it
#     """
#
#     def __init__(
#         self,
#         model,
#         warm_up_steps=100000,
#         steps=100000,
#         tune_frequency=5000,
#         discard_warm_up_data=True,
#         swap_frequency=0.3,
#         optimum_matching=True,
#         optimum_bounds='Normal'
#     ):
#         """
#         Parameters
#         ----------
#         swap_frequency: float
#             The percentage of times the simulation tries to jump between models.
#         optimum_matching: bool
#         """
#         super().__init__(model, warm_up_steps, steps, tune_frequency, discard_warm_up_data)
#
#         if not isinstance(model, ModelCollection):
#             raise ValueError('The model must be a `ModelCollection`.')
#
#         self._swap_frequency = swap_frequency
#
#         self._optimum_matching = optimum_matching
#         self._optimum_bounds = optimum_bounds
#
#     def _run_step(self, current_params, proposal_scales, current_log_prob):
#
#         proposed_params = current_params.copy()
#
#         random_move = np.random.random()
#
#         if random_move <= self._swap_frequency:
#
#             (
#                 proposed_params,
#                 proposed_log_prob,
#                 proposed_model,
#                 rjmc_jacobian,
#                 rjmc_transition,
#             ) = self.model_proposal(proposed_params)
#
#             alpha = (
#                 (proposed_log_prob - current_log_prob)
#                 + np.log(rjmc_jacobian)
#                 + np.log(rjmc_transition)
#             )
#
#             if proposed_log_prob == math.nan:
#                 # TODO: alpha = -math.inf
#                 proposed_log_prob = -math.inf
#
#         else:
#
#             proposed_params = current_params.copy()
#
#             proposed_params, proposed_log_prob = self.parameter_proposal(
#                 proposed_params, proposal_scales
#             )
#             alpha = proposed_log_prob - current_log_prob
#
#         acceptance = self._accept_reject(alpha)
#
#         if acceptance:
#
#             new_log_prob = proposed_log_prob
#             new_params = proposed_params
#
#         else:
#
#             new_log_prob = current_log_prob
#             new_params = current_params
#
#         return new_params, new_log_prob, acceptance
#
#     def model_proposal(self, prior, proposed_params, compound_2CLJ):
#
#         proposed_model = copy.deepcopy(self.current_model)
#
#         # Propose new model to jump to
#         while proposed_model == self.current_model:
#
#             proposed_model = int(np.floor(np.random.random() * self.n_models))
#
#             if proposed_model == 2 and self.current_model == 1:
#                 proposed_model = copy.deepcopy(self.current_model)
#             elif proposed_model == 1 and self.current_model == 2:
#                 proposed_model = copy.deepcopy(self.current_model)
#
#         self.lamda = 5
#         proposed_params[0] = proposed_model
#         self.w = 1
#
#         proposed_params = self.model_transition(proposed_model, proposed_params)
#
#         proposed_log_prob = self.calc_posterior(prior, compound_2CLJ, proposed_params)
#         jacobian_matrix = self.jacobian()
#         rjmc_jacobian = jacobian_matrix[self.current_model, proposed_model]
#         transition_matrix = self.transition_function()
#         rjmc_transition = transition_matrix[self.current_model, proposed_model]
#         # Return values of jacobian in order to properly calculate accept/reject
#         return (
#             proposed_params,
#             proposed_log_prob,
#             proposed_model,
#             rjmc_jacobian,
#             rjmc_transition,
#         )
#
#     def model_transition(self, proposed_model, proposed_params):
#         if proposed_model == 1 and self.current_model == 0:
#             self.move_proposals[0, 1] += 1
#             # AUA ---> AUA+Q
#             if self.optimum_matching[0] == "True":
#
#                 # Optimum Matching
#                 proposed_params[1] = (
#                     self.opt_params_AUA_Q[0] / self.opt_params_AUA[0]
#                 ) * proposed_params[1]
#                 proposed_params[2] = (
#                     self.opt_params_AUA_Q[1] / self.opt_params_AUA[1]
#                 ) * proposed_params[2]
#                 proposed_params[3] = (
#                     self.opt_params_AUA_Q[2] / self.opt_params_AUA[2]
#                 ) * proposed_params[3]
#
#             self.w = np.random.random()
#
#             # THIS IS IMPORTANT needs to be different depending on which direction
#
#             proposed_params[4] = -(1 / self.lamda) * np.log(self.w)
#             # Propose a value of Q from an exponential distribution using the inverse
#             # CDF method (this is nice because it keeps the transition probability
#             # simple)
#
#         elif proposed_model == 0 and self.current_model == 1:
#             self.move_proposals[1, 0] += 1
#             # AUA+Q ----> AUA
#
#             if self.optimum_matching[0] == "True":
#                 # Optimum Matching
#                 proposed_params[1] = (
#                     self.opt_params_AUA[0] / self.opt_params_AUA_Q[0]
#                 ) * proposed_params[1]
#                 proposed_params[2] = (
#                     self.opt_params_AUA[1] / self.opt_params_AUA_Q[1]
#                 ) * proposed_params[2]
#                 proposed_params[3] = (
#                     self.opt_params_AUA[2] / self.opt_params_AUA_Q[2]
#                 ) * proposed_params[3]
#
#             # w=params[4]/2
#
#             # Still need to calculate what "w" (dummy variable) would be even though
#             # we don't use it (to satisfy detailed balance)
#             self.w = np.exp(-self.lamda * proposed_params[4])
#
#             proposed_params[4] = 0
#
#         elif proposed_model == 2 and self.current_model == 0:
#             self.move_proposals[0, 2] += 1
#
#             # AUA--->UA
#
#             proposed_params[1] = (
#                 self.opt_params_UA[0] / self.opt_params_AUA[0]
#             ) * proposed_params[1]
#             proposed_params[2] = (
#                 self.opt_params_UA[1] / self.opt_params_AUA[1]
#             ) * proposed_params[2]
#             proposed_params[3] = self.opt_params_UA[2]
#
#             proposed_params[4] = 0
#             self.w = 1
#
#         elif proposed_model == 0 and self.current_model == 2:
#             # UA ----> AUA
#             self.move_proposals[2, 0] += 1
#
#             proposed_params[1] = (
#                 self.opt_params_AUA[0] / self.opt_params_UA[0]
#             ) * proposed_params[1]
#             proposed_params[2] = (
#                 self.opt_params_AUA[1] / self.opt_params_UA[1]
#             ) * proposed_params[2]
#             proposed_params[3] = (
#                 self.opt_params_AUA[2] / self.opt_params_UA[2]
#             ) * proposed_params[3]
#             self.w = 1
#             proposed_params[4] = 0
#         return proposed_params
#
#     def refit_prior(self, prior_type):
#         if prior_type == "exponential":
#             loc, scale = utils.fit_exponential_sp(self.trace_model_1)
#             new_prior = (0, scale)
#
#             Q_prior = [prior_type, new_prior]
#         elif prior_type == "gamma":
#             alpha, loc, scale = utils.fit_gamma_sp(self.trace_model_1)
#             new_prior = (alpha, loc, scale)
#             Q_prior = [prior_type, new_prior]
#         else:
#             raise ValueError("Prior type not implemented")
#         return Q_prior

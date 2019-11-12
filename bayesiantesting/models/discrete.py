"""
A collection of model collections i.e collections
of linked models which we wish to sample between
either with techniques such as RJMC, or to employ
free-energy like techniques to evaluate the individual
model evidences.

Models in this module should inherit from the `ModelCollection`
subclass.
"""

# class TwoCenterLJModelCollection(ModelCollection):
#
#     def __init__(self, models):
#
#         assert all(isinstance(model, TwoCenterLJModel) for model in models)
#         super().__init__(models)
#
#     def gen_Tmatrix(self, prior, compound_2CLJ):
#         """ Generate Transition matrices based on the optimal eps, sig, Q for different models"""
#
#         # Currently this is not used for moves between AUA and AUA+Q, because it
#         # doesn't seem to help.  Still used for UA and AUA moves
#
#         def obj_AUA(eps_sig_Q):
#             return -self.calc_posterior(
#                 prior,
#                 compound_2CLJ,
#                 [0, eps_sig_Q[0], eps_sig_Q[1], eps_sig_Q[2], eps_sig_Q[3]],
#             )
#
#         def obj_AUA_Q(eps_sig_Q):
#             return -self.calc_posterior(
#                 prior,
#                 compound_2CLJ,
#                 [1, eps_sig_Q[0], eps_sig_Q[1], eps_sig_Q[2], eps_sig_Q[3]],
#             )
#
#         def obj_2CLJ(eps_sig_Q):
#             return -self.calc_posterior(
#                 prior,
#                 compound_2CLJ,
#                 [2, eps_sig_Q[0], eps_sig_Q[1], eps_sig_Q[2], eps_sig_Q[3]],
#             )
#
#         guess_0 = [0, *self.ff_params_ref[1]]
#         guess_1 = [1, *self.ff_params_ref[0]]
#         guess_2 = [2, *self.ff_params_ref[2]]
#         guess_2[3] = self.NIST_bondlength
#
#         guess_AUA = [guess_0[1], guess_0[2], guess_0[3], guess_0[4]]
#         guess_AUA_Q = [guess_1[1], guess_1[2], guess_1[3], guess_1[4]]
#         guess_UA = [guess_2[1], guess_2[2], guess_2[3], guess_2[4]]
#
#         # Make sure bounds are in a reasonable range so that models behave properly
#         if self.optimum_bounds == "Normal":
#             bnd_AUA = (
#                 (0.85 * guess_AUA[0], guess_AUA[0] * 1.15),
#                 (0.9 * guess_AUA[1], guess_AUA[1] * 1.1),
#                 (0.9 * guess_AUA[2], guess_AUA[2] * 1.1),
#                 (0, 0),
#             )
#             bnd_AUA_Q = (
#                 (0.85 * guess_AUA_Q[0], guess_AUA_Q[0] * 1.15),
#                 (0.9 * guess_AUA_Q[1], guess_AUA_Q[1] * 1.1),
#                 (0.9 * guess_AUA_Q[2], guess_AUA_Q[2] * 1.1),
#                 (0.9 * guess_AUA_Q[3], guess_AUA_Q[3] * 1.1),
#             )
#             bnd_UA = (
#                 (0.85 * guess_UA[0], guess_UA[0] * 1.15),
#                 (0.9 * guess_UA[1], guess_UA[1] * 1.1),
#                 (1 * guess_UA[2], guess_UA[2] * 1),
#                 (0.90 * guess_UA[3], guess_UA[3] * 1.1),
#             )
#         elif self.optimum_bounds == "Expanded":
#
#             bnd_AUA = (
#                 (0 * guess_AUA[0], guess_AUA[0] * 2),
#                 (0 * guess_AUA[1], guess_AUA[1] * 2),
#                 (0.5 * guess_AUA[2], guess_AUA[2] * 2),
#                 (0, 0),
#             )
#             bnd_AUA_Q = (
#                 (0 * guess_AUA_Q[0], guess_AUA_Q[0] * 2),
#                 (0 * guess_AUA_Q[1], guess_AUA_Q[1] * 2),
#                 (0 * guess_AUA_Q[2], guess_AUA_Q[2] * 2),
#                 (0 * guess_AUA_Q[3], guess_AUA_Q[3] * 2),
#             )
#             bnd_UA = (
#                 (0.85 * guess_UA[0], guess_UA[0] * 1.15),
#                 (0.9 * guess_UA[1], guess_UA[1] * 1.1),
#                 (1 * guess_UA[2], guess_UA[2] * 1),
#                 (0.90 * guess_UA[3], guess_UA[3] * 1.1),
#             )
#         else:
#             raise ValueError('optimum_bounds must be "Normal" or "Expanded"')
#         # Help debug
#         #    print(bnd_LJ)
#         #    print(bnd_UA)
#         #    print(bnd_AUA)
#
#         opt_AUA = minimize(obj_AUA, guess_AUA, bounds=bnd_AUA)
#         opt_AUA_Q = minimize(obj_AUA_Q, guess_AUA_Q, bounds=bnd_AUA_Q)
#         opt_UA = minimize(obj_2CLJ, guess_UA, bounds=bnd_UA)
#         # Help debug
#         #    print(opt_LJ)
#         #    print(opt_UA)
#         #    print(opt_AUA)
#
#         self.opt_params_AUA = opt_AUA.x[0], opt_AUA.x[1], opt_AUA.x[2], opt_AUA.x[3]
#         self.opt_params_AUA_Q = (
#             opt_AUA_Q.x[0],
#             opt_AUA_Q.x[1],
#             opt_AUA_Q.x[2],
#             opt_AUA_Q.x[3],
#         )
#         self.opt_params_UA = opt_UA.x[0], opt_UA.x[1], opt_UA.x[2], opt_UA.x[3]
#
#     def get_initial_state(self):
#
#         # rnorm = np.random.normal
#         #
#         # initial_values[0] = random.randint(0, self.n_models - 1)
#         #
#         # if initial_model == "AUA":
#         #     initial_values[0] = 0
#         # elif initial_model == "AUA+Q":
#         #     initial_values[0] = 1
#         # elif initial_model == "UA":
#         #     initial_values[0] = 2
#         #
#         # if initial_values[0] == "AUA":
#         #
#         #     initial_values[1] = rnorm(self.opt_params_AUA[0], self.opt_params_AUA[0] / 20)
#         #     initial_values[2] = rnorm(self.opt_params_AUA[1], self.opt_params_AUA[1] / 20)
#         #     initial_values[3] = rnorm(self.opt_params_AUA[2], self.opt_params_AUA[2] / 20)
#         #     initial_values[4] = 0
#         #
#         # elif initial_values[0] == "AUA+Q":
#         #
#         #     initial_values[1] = rnorm(self.opt_params_AUA_Q[0], self.opt_params_AUA_Q[0] / 20)
#         #     initial_values[2] = rnorm(self.opt_params_AUA_Q[1], self.opt_params_AUA_Q[1] / 20)
#         #     initial_values[3] = rnorm(self.opt_params_AUA_Q[2], self.opt_params_AUA_Q[2] / 20)
#         #     initial_values[4] = rnorm(self.opt_params_AUA_Q[2], self.opt_params_AUA_Q[2] / 20)
#         #
#         # elif initial_values[0] == "UA":
#         #
#         #     initial_values[1] = rnorm(self.opt_params_UA[0], self.opt_params_UA[0] / 20)
#         #     initial_values[2] = rnorm(self.opt_params_UA[1], self.opt_params_UA[1] / 20)
#         #     initial_values[3] = self.NIST_bondlength
#         #     initial_values[4] = 0
#         pass
#
#     def transition_function(self):
#
#         unif = distributions.uniform.pdf
#
#         transition_matrix = np.ones((self.model.n_models, self.model.n_models))
#
#         # These are proposal distributions for "new" variables (that exist in one
#         # model but not the other).  They have been cleverly chosen to all equal 1
#         g_0_1 = unif(self.w, 0, 1)
#         g_1_0 = 1
#         g_0_2 = 1
#         g_2_0 = 1
#
#         # These are probabilities of proposing a model from one model to another.
#         # The probability is half for moves originating in AUA because they can
#         # move either to UA or AUA+Q. We disallow moves between UA and AUA+Q
#         # directly
#         q_0_1 = 1 / 2
#         q_1_0 = 1
#         q_0_2 = 1 / 2
#         q_2_0 = 1
#
#         # Note that this is really times swap_freq but that term always cancels.
#         transition_matrix[0, 1] = g_1_0 * q_1_0 / (g_0_1 * q_0_1)
#         transition_matrix[1, 0] = g_0_1 * q_0_1 / (g_1_0 * q_1_0)
#         transition_matrix[0, 2] = g_2_0 * q_2_0 / (g_0_2 * q_0_2)
#         transition_matrix[2, 0] = g_0_2 * q_0_2 / (g_2_0 * q_2_0)
#
#         return transition_matrix
#
#     def jacobian(self):
#
#         jacobian_matrix = np.ones((self.n_models, self.n_models))
#
#         if self.optimum_matching == "True":
#             jacobian_matrix[0, 1] = (
#                 (1 / (self.lamda * self.w))
#                 * (
#                     self.opt_params_AUA_Q[0]
#                     * self.opt_params_AUA_Q[1]
#                     * self.opt_params_AUA_Q[2]
#                 )
#                 / (
#                     self.opt_params_AUA[0]
#                     * self.opt_params_AUA[1]
#                     * self.opt_params_AUA[2]
#                 )
#             )
#             jacobian_matrix[1, 0] = (
#                 self.lamda
#                 * (
#                     self.opt_params_AUA[0]
#                     * self.opt_params_AUA[1]
#                     * self.opt_params_AUA[2]
#                 )
#                 / (
#                     self.opt_params_AUA_Q[0]
#                     * self.opt_params_AUA_Q[1]
#                     * self.opt_params_AUA_Q[2]
#                 )
#             )
#             jacobian_matrix[0, 1] = 1 / (self.lamda * self.w)
#             jacobian_matrix[1, 0] = self.w * self.lamda
#         else:
#             jacobian_matrix[0, 1] = 1 / (self.lamda * self.w)
#             jacobian_matrix[1, 0] = self.w * self.lamda
#
#         # Optimum Matching for UA --> AUA
#         # jacobian[0,1]=(1/(lamda*w))*(opt_params_AUA_Q[0]*opt_params_AUA_Q[1]*
#         # opt_params_AUA_Q[2])/(opt_params_AUA[0]*opt_params_AUA[1]*opt_params_AUA[2])
#         # jacobian[1,0]=lamda*(opt_params_AUA[0]*opt_params_AUA[1]*opt_params_AUA[2])
#         # /(opt_params_AUA_Q[0]*opt_params_AUA_Q[1]*opt_params_AUA_Q[2])
#         jacobian_matrix[0, 2] = (
#             self.opt_params_UA[0] * self.opt_params_UA[1] * self.opt_params_UA[2]
#         ) / (self.opt_params_AUA[0] * self.opt_params_AUA[1] * self.opt_params_AUA[2])
#         jacobian_matrix[2, 0] = (
#             self.opt_params_AUA[0] * self.opt_params_AUA[1] * self.opt_params_AUA[2]
#         ) / (self.opt_params_UA[0] * self.opt_params_UA[1] * self.opt_params_UA[2])
#         # Direct transfer for AUA->AUA+Q
#
#         return jacobian_matrix

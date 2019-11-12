import numpy as np
import torch.distributions


class Model:
    """ Sets up a simply model based on the user-specified prior
    types and parameters
    """

    @property
    def n_trainable_parameters(self):
        """int: The number of trainable parameters within this model."""
        return len(self._prior_labels)

    @property
    def trainable_parameter_labels(self):
        """list of str: The friendly names of the parameters which are allowed to vary."""
        return self._prior_labels

    @property
    def n_total_parameters(self):
        """int: The total number of parameters within this model."""
        return len(self._all_parameter_labels)

    @property
    def all_parameter_labels(self):
        """list of str: The friendly names of the parameters within this model."""
        return len(self._all_parameter_labels)

    def __init__(self, prior_settings):
        """Constructs a new `MCMCModel` object.

        Parameters
        ----------
        prior_settings: dict of str and tuple of float
            The settings for each of the priors. There should be
            one entry per parameter.
        """
        self._priors = []
        self._prior_labels = []

        for prior_name in prior_settings:

            self._priors.append(self._initialize_prior(prior_settings[prior_name]))
            self._prior_labels.append(prior_name)

        self._all_parameter_labels = [*self._prior_labels]

    @staticmethod
    def _initialize_prior(settings):

        prior_type, prior_values = settings

        if prior_type == "exponential":

            if not np.isclose(prior_values[0], 0.0):
                # The loc argument is not supported in PyTorch.
                raise NotImplementedError()

            prior = torch.distributions.Exponential(rate=1.0 / prior_values[1])

        elif prior_type == "gamma":

            if not np.isclose(prior_values[1], 0.0):
                # The loc argument is not supported in PyMC3.
                raise NotImplementedError()

            prior = torch.distributions.Gamma(
                prior_values[0], rate=1.0 / prior_values[2]
            )

        else:
            raise NotImplementedError()

        return prior

    def sample_priors(self):
        """Generates a set of random parameters from the prior
        distributions. Those parameters without a prior will be
        assigned a value of 0.

        Returns
        -------
        numpy.ndarray:
            The sampled parameters with shape=(`n_total_parameters`).
        """

        initial_parameters = np.zeros(self.n_total_parameters)

        for index, prior in enumerate(self._priors):
            initial_parameters[index] = prior.rsample()

        return initial_parameters

    def evaluate_log_prior(self, parameters):
        """Evaluates the log value of the prior for a
        set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=(n parameters, 1))
            to evaluate at.

        Returns
        -------
        float
            The sum of the log values of priors evaluated at `parameters`.
        """
        return sum(
            [
                prior.log_prob(parameters[index]).item()
                for index, prior in enumerate(self._priors)
            ]
        )

    def evaluate_log_likelihood(self, parameters):
        """Evaluates the log value of the this models likelihood for
        a set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=(n parameters, 1))
            to evaluate at.

        Returns
        -------
        float
            The log value of the likelihood evaluated at `parameters`.
        """
        raise NotImplementedError()

    def evaluate_log_posterior(self, parameters):
        """Evaluates the *unnormalized* log posterior for
        a set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=(n parameters, 1))
            to evaluate at.

        Returns
        -------
        float
            The log value of the posterior evaluated at `parameters`.
        """
        return self.evaluate_log_prior(parameters) + self.evaluate_log_likelihood(
            parameters
        )


class ModelCollection:
    """Represents a collection of models to simultaneously optimize.
    """

    @property
    def models(self):
        """tuple of Model: The models which belong to this collection."""
        return self._models

    @property
    def n_models(self):
        """int: The number models which belong to this collection."""
        return len(self._models)

    def __init__(self, models):
        """Initializes self.

        Parameters
        ----------
        models: list of Model
            The models which belong to this collection.
        """
        self._models = tuple(*models)
        raise NotImplementedError()

    def __len__(self):
        return self.n_models


class TwoCenterLennardJones(Model):
    """A representation of the two-center Lennard-Jones model, which
    can be evaluated using a surrogate model against a `NISTDataSet`.
    """

    @property
    def total_parameters(self):
        return 4

    def __init__(
        self, prior_settings, reference_data_set, property_types, surrogate_model
    ):
        """Constructs a new `TwoCenterLennardJones` model.

        Parameters
        ----------
        reference_data_set: NISTDataSet
            The data set which contains the experimentally observed values
            that this model is attempts to reproduce.
        property_types: list of NISTDataType
            The list of property types which this model attempts
            to reproduce.
        surrogate_model: SurrogateModel
            The surrogate model to use when evaluating the likelihood function.
        """
        super().__init__(prior_settings)

        self._all_parameter_labels = ["epsilon", "sigma", "L", "Q"]

        if "epsilon" not in self._prior_labels or "sigma" not in self._prior_labels:

            raise ValueError(
                "Both an `epsilon` and `sigma` prior must be provided."
                "The `L` and `Q` parameters are optional."
            )

        for parameter_label in self._prior_labels:

            if parameter_label in self._all_parameter_labels:
                continue

            raise ValueError(
                f"The only allowed parameters of this model are {', '.join(self._all_parameter_labels)}. "
                f"The `L` and `Q` parameters are optional."
            )

        self._property_types = property_types

        self._reference_data = {}
        self._reference_precisions = {}

        for property_type in self._property_types:

            self._reference_data[property_type] = np.asarray(
                reference_data_set.get_data(property_type)
            )
            self._reference_precisions[property_type] = np.asarray(
                reference_data_set.get_precision(property_type)
            )

        self._surrogate_model = surrogate_model

    def evaluate_log_likelihood(self, parameters):
        """Evaluates the log value of the this models likelihood for
        a set of parameters. based on a gaussian penalty function.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=(n parameters, 1))
            to evaluate at.

        Returns
        -------
        float
            The log value of the likelihood evaluated at `parameters`.
        """
        log_p = 0.0

        for property_type in self._property_types:

            reference_data = self._reference_data[property_type]
            precisions = self._reference_precisions[property_type]

            temperatures = reference_data[:, 0]

            reference_values = reference_data[:, 1]
            surrogate_values = self._surrogate_model.evaluate(
                property_type, parameters, temperatures
            )

            sm_torch = torch.from_numpy(surrogate_values)
            prec_torch = torch.from_numpy(precisions) ** -2.0
            ref_torch = torch.from_numpy(reference_values)

            # Compute likelihood based on gaussian penalty function
            log_p += torch.sum(
                torch.distributions.Normal(sm_torch, prec_torch).log_prob(ref_torch)
            ).item()

            # log_p += sum(
            #     pymc3.distributions.Normal.dist(mu=mu, sigma=precision ** -2.0)
            #     .logp(x)
            #     .eval()
            #     for x, mu, precision in zip(
            #         reference_values, surrogate_values, precisions
            #     )
            # )

        return log_p


# class TwoCenterLJModelCollection(ModelCollection):
#
#     def __init__(self, models):
#
#         assert all(isinstance(model, TwoCenterLennardJones) for model in models)
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

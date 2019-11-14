import autograd
import numpy
import numpy as np
import torch.distributions
from bayesiantesting.utils import distributions


class Model:
    """ Sets up a simply model based on the user-specified prior
    types and parameters
    """

    @property
    def name(self):
        """str: The name of this model."""
        return self._name

    @property
    def priors(self):
        return self._priors

    @property
    def fixed_parameters(self):
        return self._fixed_parameters

    @property
    def n_trainable_parameters(self):
        """int: The number of trainable parameters within this model."""
        return len(self._prior_labels)

    @property
    def trainable_parameter_labels(self):
        """list of str: The friendly names of the parameters which are allowed to vary."""
        return self._prior_labels

    @property
    def n_fixed_parameters(self):
        """int: The number of fixed parameters within this model."""
        return len(self._fixed_labels)

    @property
    def fixed_parameter_labels(self):
        """list of str: The friendly names of the parameters which are fixed."""
        return self._fixed_labels

    @property
    def n_total_parameters(self):
        """int: The total number of parameters within this model."""
        return self.n_trainable_parameters + self.n_fixed_parameters

    @property
    def all_parameter_labels(self):
        """list of str: The friendly names of the parameters within this model."""
        return self._prior_labels + self._fixed_labels

    def __init__(self, name, priors, fixed_parameters):
        """Constructs a new `MCMCModel` object.

        Parameters
        ----------
        name: str
            The name of this model.
        priors: dict of str and tuple of float
            The settings for each of the priors, whose keys are the friendly
            name of the parameter associated with the prior. There should be
            one entry per trainable parameter.
        fixed_parameters: dict of str and float
            The values of the fixed model parameters, whose keys of the name
            associated with the parameter.
        """
        self._name = name

        self._priors = []
        self._prior_labels = []

        self._fixed_parameters = []
        self._fixed_labels = []

        for parameter_name in priors:

            self._priors.append(self._initialize_prior(priors[parameter_name]))
            self._prior_labels.append(parameter_name)

        for parameter_name in fixed_parameters:

            self._fixed_parameters.append(fixed_parameters[parameter_name])
            self._fixed_labels.append(parameter_name)

        common_parameters = set(self._fixed_labels).intersection(
            set(self._prior_labels)
        )

        if len(common_parameters) > 0:

            raise ValueError(
                f"The {', '.join(common_parameters)} have been flagged "
                f"as being both fixed and trainable."
            )

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
                # The loc argument is not supported in PyTorch.
                raise NotImplementedError()

            prior = torch.distributions.Gamma(
                prior_values[0], rate=1.0 / prior_values[2]
            )

        elif prior_type == "normal":

            prior = torch.distributions.Normal(prior_values[0], prior_values[1])

        else:
            raise NotImplementedError()

        return prior

    def sample_priors(self):
        """Generates a set of random parameters from the prior
        distributions. Those parameters without a prior will be
        assigned their fixed values.

        Returns
        -------
        numpy.ndarray:
            The sampled parameters with shape=(`n_total_parameters`).
        """

        initial_parameters = np.zeros(self.n_total_parameters)

        for index, prior in enumerate(self._priors):
            initial_parameters[index] = prior.rsample().item()

        for index, parameter in enumerate(self._fixed_parameters):
            initial_parameters[index + self.n_trainable_parameters] = parameter

        return initial_parameters

    def evaluate_log_prior(self, parameters):
        """Evaluates the log value of the prior for a
        set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=n_parameters)
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
            The values of the parameters (with shape=n_parameters)
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
            The values of the parameters (with shape=n_parameters)
            to evaluate at.

        Returns
        -------
        float
            The log value of the posterior evaluated at `parameters`.
        """
        return self.evaluate_log_prior(parameters) + self.evaluate_log_likelihood(
            parameters
        )

    def compute_percentage_deviations(self, parameters):
        """Computes the deviation of this models predicted
        values from the measured data it is being conditioned
        upon.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=n_parameters)
            to evaluate at.

        Returns
        -------
        dict of str and numpy.ndarray

        """
        raise NotImplementedError()


class ModelCollection:
    """Represents a collection of models to simultaneously optimize.
    """

    @property
    def name(self):
        """str: The name of this model."""
        return self._name

    @property
    def models(self):
        """tuple of Model: The models which belong to this collection."""
        return self._models

    @property
    def n_models(self):
        """int: The number models which belong to this collection."""
        return len(self._models)

    def __init__(self, name, models):
        """Initializes self.

        Parameters
        ----------
        name: str
            The name of this collection.
        models: List of Model
            The models which belong to this collection.
        """

        # Make sure there are no models with duplicate names.
        assert len(set(model.name for model in models)) == len(models)

        self._name = name
        self._models = tuple(models)

        for model in self._models:

            if all(
                isinstance(prior, torch.distributions.Exponential)
                for prior in model.priors
            ):
                continue

            raise ValueError("Currently only exponential priors are supported.")

    def _mapping_function(
        self, parameter, model_index_a, model_index_b, parameter_index
    ):

        model_a = self._models[model_index_a]
        model_b = self._models[model_index_b]

        if (
            parameter_index >= model_a.n_trainable_parameters
            and parameter_index >= model_b.n_trainable_parameters
        ):

            # These parameters aren't being trained so we don't need to
            # do any mapping so long as both models take the same fixed
            # value.
            if not numpy.isclose(
                model_a.fixed_parameters[
                    parameter_index - model_a.n_trainable_parameters
                ],
                model_b.fixed_parameters[
                    parameter_index - model_b.n_trainable_parameters
                ],
            ):

                raise NotImplementedError()

            return parameter

        elif (
            parameter_index < model_a.n_trainable_parameters
            and parameter_index < model_b.n_trainable_parameters
        ):

            prior_0_rate = model_a.priors[parameter_index].rate.item()
            prior_1_rate = model_b.priors[parameter_index].rate.item()

            cdf_x = distributions.exponential_cdf(parameter, prior_0_rate)
            return distributions.exponential_inverse_cdf(cdf_x, prior_1_rate)

        elif (
            model_a.n_trainable_parameters
            > parameter_index
            >= model_b.n_trainable_parameters
        ):

            # Handle the case where we are mapping to a model with a lower dimension.
            prior_0_rate = model_a.priors[parameter_index].rate.item()
            return distributions.exponential_cdf(parameter, prior_0_rate)

        elif (
            model_a.n_trainable_parameters
            <= parameter_index
            < model_b.n_trainable_parameters
        ):

            # Handle the case where we are mapping to a model with a higher dimension.
            prior_1_rate = model_b.priors[parameter_index].rate.item()
            return distributions.exponential_inverse_cdf(parameter, prior_1_rate)

        raise NotImplementedError()

    def map_parameters(self, parameters, model_index_a, model_index_b):

        current_parameters = parameters.copy()

        new_parameters = numpy.empty(parameters.shape)
        jacobians = numpy.empty(parameters.shape)

        n_parameters = max(
            self._models[model_index_a].n_total_parameters,
            self._models[model_index_b].n_total_parameters,
        )

        jacobian_function = autograd.grad(self._mapping_function)

        if (
            self._models[model_index_a].n_trainable_parameters
            < self._models[model_index_b].n_trainable_parameters
        ):

            # If we are moving to a higher dimensional model, we
            # set the 'ghost' parameters to a random number drawn
            # from a uniform distribution.
            for j in range(
                self._models[model_index_a].n_trainable_parameters,
                self._models[model_index_b].n_trainable_parameters,
            ):

                current_parameters[j] = torch.rand((1,)).item()

        for i in range(n_parameters):

            new_parameters[i] = self._mapping_function(
                current_parameters[i], model_index_a, model_index_b, i
            )
            jacobians[i] = jacobian_function(
                current_parameters[i], model_index_a, model_index_b, i
            )

        return current_parameters, new_parameters, jacobians

    def transition_probabilities(self, model_index_a, model_index_b):
        return 1.0

    def evaluate_log_prior(self, model_index, parameters):
        """Evaluates the log value of the prior for a
        given model and corresponding parameters.

        Parameters
        ----------
        model_index: int
            The index of the model to evaluate.
        parameters: numpy.ndarray
            The values of the parameters (with shape=n_parameters)
            to evaluate at.

        Returns
        -------
        float
            The sum of the log values of priors evaluated at `parameters`.
        """
        return self._models[model_index].evaluate_log_prior(parameters)

    def evaluate_log_likelihood(self, model_index, parameters):
        """Evaluates the log value of the this models likelihood for a
        given model and corresponding parameters.

        Parameters
        ----------
        model_index: int
            The index of the model to evaluate.
        parameters: numpy.ndarray
            The values of the parameters (with shape=n_parameters)
            to evaluate at.

        Returns
        -------
        float
            The log value of the likelihood evaluated at `parameters`.
        """
        return self._models[model_index].evaluate_log_likelihood(parameters)

    def evaluate_log_posterior(self, model_index, parameters):
        """Evaluates the *unnormalized* log posterior for a
        given model and corresponding parameters.

        Parameters
        ----------
        model_index: int
            The index of the model to evaluate.
        parameters: numpy.ndarray
            The values of the parameters (with shape=n_parameters)
            to evaluate at.

        Returns
        -------
        float
            The log value of the posterior evaluated at `parameters`.
        """
        return self._models[model_index].evaluate_log_posterior(parameters)

    def __len__(self):
        return self.n_models

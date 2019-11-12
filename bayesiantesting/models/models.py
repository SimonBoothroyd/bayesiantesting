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

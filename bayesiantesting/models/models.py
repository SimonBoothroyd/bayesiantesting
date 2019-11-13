import numpy as np
import torch.distributions


class Model:
    """ Sets up a simply model based on the user-specified prior
    types and parameters
    """

    @property
    def name(self):
        """str: The name of this model."""
        return self._name

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

            prior = torch.distributions.Normal(
                prior_values[0], prior_values[1]
            )

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
            initial_parameters[index] = prior.rsample()

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

        self.__name = name
        self._models = tuple(*models)

    def __len__(self):
        return self.n_models

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

import numpy as np
import pymc3.distributions


class Model:
    """ Sets up a simply model based on the user-specified prior types and parameters

    Warnings
    --------
    This is a simple placeholder until a cleaner solution is
    implemented.
    """

    def __init__(self, prior_settings):
        """Constructs a new `MCMCModel` object.

        Parameters
        ----------
        prior_settings: dict of str and tuple of float
            The settings for each of the parameter priors.
        """
        self.priors = []
        self.prior_labels = []

        for prior_name in prior_settings:

            self.priors.append(self._initialize_prior(prior_settings[prior_name]))
            self.prior_labels.append(prior_name)

        self._initial_parameters = None

    @staticmethod
    def _initialize_prior(settings):

        prior_type, prior_values = settings

        if prior_type == "exponential":

            if not np.isclose(prior_values[0], 0.0):
                # The loc argument is not supported in PyMC3.
                raise NotImplementedError()

            prior = pymc3.distributions.Exponential.dist(lam=1.0 / prior_values[1])

        elif prior_type == "gamma":

            if not np.isclose(prior_values[1], 0.0):
                # The loc argument is not supported in PyMC3.
                raise NotImplementedError()

            prior = pymc3.distributions.Gamma(
                alpha=prior_values[0], beta=1.0 / prior_values[2]
            )

        else:
            raise NotImplementedError()

        return prior

    def sample_priors(self):
        """Generates a set of random parameters from the prior
        distributions.

        Returns
        -------
        numpy.ndarray:
            The sampled parameters.
        """

        initial_parameters = np.zeros((len(self.priors), 1))

        for index, prior in enumerate(self.priors):
            initial_parameters[index] = prior.random()

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
                self.priors[index].logp(parameters[index]).eval()
                for index in range(len(self.priors))
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

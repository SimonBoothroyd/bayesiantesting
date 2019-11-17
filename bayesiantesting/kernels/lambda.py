"""
Code to perform 'alchemical' like 'lambda scaling free energy'
calculations to estimate Bayes factors. In these cases, lambda
is a hyperparameter which interpolates between the prior and
posterior distributions.
"""
import math

import numpy as np
import torch
from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.models.models import ModelCollection


class LambdaSimulation(MCMCSimulation):
    """Builds an object that runs an MCMC simulation at a specific value of
    lambda - a hyperparameter which interpolates between the prior and posterior
    distributions.

    The samples will be generated from the following simple distribution:

        p(x|D, λ) = (1-λ)*p(x) + λ*p(x|D)

    where p(x) is the prior on x, and p(x|D) is the posterior distribution
    given by

        p(x|D) = p(x)p(x|D)

    At λ=0.0 only the prior is sampled, at λ=1.0 the full prior is sampled.
    """

    def __init__(
        self,
        model_collection,
        warm_up_steps=100000,
        steps=100000,
        tune_frequency=5000,
        discard_warm_up_data=True,
        lambda_value=1.0,
    ):
        """
        Parameters
        ----------
        lambda_value: float
            The value of lambda to sample at.
        """

        super().__init__(
            model_collection, warm_up_steps, steps, tune_frequency, discard_warm_up_data
        )

        self._lambda = lambda_value

    def _evaluate_log_p(self, parameters, model_index):

        model = self._model_collection.models[model_index]

        prior_weight = (1.0 - self._lambda) * model.evaluate_log_prior(parameters)
        posterior_weight = self._lambda * model.evaluate_log_posterior(parameters)

        return prior_weight + posterior_weight

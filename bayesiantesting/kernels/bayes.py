"""
Code to perform 'alchemical' like 'lambda scaling free energy'
calculations to estimate Bayes factors. In these cases, lambda
is a hyperparameter which interpolates between the prior and
posterior distributions.
"""
import functools
from multiprocessing.pool import ThreadPool

import numpy

from bayesiantesting.kernels import MCMCSimulation


class LambdaSimulation(MCMCSimulation):
    """Builds an object that runs an MCMC simulation at a specific value of
    lambda - a hyperparameter which interpolates between the prior and posterior
    distributions.

    The samples will be generated from the following simple distribution:

        ln p(x|D, λ) = (1-λ) * ln p(x) + λ * ln p(x|D)

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


class ThermodynamicIntegration:
    def __init__(
        self,
        legendre_gauss_degree,
        model,
        warm_up_steps=100000,
        steps=100000,
        tune_frequency=5000,
        discard_warm_up_data=True,
    ):
        """
        Parameters
        ----------
        legendre_gauss_degree: int
            The number of lambdas to use for the
            Gauss-Legendre quadrature integration.
        model: Model
            The model whose bayes factors should be computed.
        warm_up_steps: int
            The number of warm-up steps to take when simulating at
            each lambda. During this time all move proposals will
            be tuned.
        steps: int
            The number of steps to simulate at each value of lambda for.
        tune_frequency: int
            The frequency with which to tune the move proposals.
        discard_warm_up_data: bool
            If true, all data generated during the warm-up period will
            be discarded.
        """

        self._model = model
        self._warm_up_steps = warm_up_steps
        self._steps = steps
        self._tune_frequency = tune_frequency
        self._discard_warm_up_data = discard_warm_up_data

        # Choose the lambda values
        lambda_values, lambda_weights = numpy.polynomial.legendre.leggauss(legendre_gauss_degree)

        self._lambda_values = lambda_values * 0.5 + 0.5
        self._lambda_weights = lambda_weights * 0.5

    def _validate_parameter_shapes(self, initial_parameters):

        if len(initial_parameters) != self._model.n_trainable_parameters:

            raise ValueError(
                f"The initial parameters vector should contain "
                f"one value for each of the trainable model parameters."
            )

    def run(self, initial_parameters, number_of_threads=1):

        # Make sure the parameters are the correct shape for the
        # specified model.
        self._validate_parameter_shapes(initial_parameters)

        # Simulate in each lambda window.
        with ThreadPool(number_of_threads) as pool:

            run_with_args = functools.partial(ThermodynamicIntegration._run_window,
                                              self._model,
                                              self._warm_up_steps,
                                              self._steps,
                                              self._tune_frequency,
                                              self._discard_warm_up_data,
                                              initial_parameters)

            results = pool.map(run_with_args, self._lambda_values)

        return results

    @staticmethod
    def _run_window(
        model,
        warm_up_steps,
        steps,
        tune_frequency,
        discard_warm_up_data,
        initial_parameters,
        lambda_value
    ):

        simulation = LambdaSimulation(
            model_collection=model,
            warm_up_steps=warm_up_steps,
            steps=steps,
            tune_frequency=tune_frequency,
            discard_warm_up_data=discard_warm_up_data,
            lambda_value=lambda_value
        )

        trace, log_p_trace, _ = simulation.run(initial_parameters, 0)

        # TODO: Thin data / equilibration detection?
        d_lop_p_d_lambda = numpy.empty(log_p_trace.shape)

        for index in range(trace.shape[0]):

            # TODO: Vectorize this.
            d_lop_p_d_lambda[index] = (
                model.evaluate_log_posterior(trace[index][1:]) -
                model.evaluate_log_prior(trace[index][1:])
            )

        return trace, log_p_trace, d_lop_p_d_lambda

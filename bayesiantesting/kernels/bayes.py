"""
Code to perform 'alchemical' like 'lambda scaling free energy'
calculations to estimate Bayes factors. In these cases, lambda
is a hyperparameter which interpolates between the prior and
posterior distributions.
"""
import functools
from multiprocessing.pool import Pool

import numpy
from bayesiantesting.kernels import MCMCSimulation
from pymbar import timeseries


class LambdaSimulation(MCMCSimulation):
    """Builds an object that runs an MCMC simulation at a specific value of
    lambda - a hyperparameter which interpolates between the prior and posterior
    distributions.

    The samples will be generated from the following simple distribution:

        p(x|D, λ) = p(x) * p(D|x)^lambda

    or rather

        ln p(x|D, λ) = ln p(x) + λ ln p(D|x)

    where p(x) is the prior on x, and p(D|x) is the likelihood distribution
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
        return model.evaluate_log_prior(
            parameters
        ) + self._lambda * model.evaluate_log_likelihood(parameters)


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
        lambda_values, lambda_weights = numpy.polynomial.legendre.leggauss(
            legendre_gauss_degree
        )

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
        with Pool(number_of_threads) as pool:

            run_with_args = functools.partial(
                ThermodynamicIntegration._run_window,
                self._model,
                self._warm_up_steps,
                self._steps,
                self._tune_frequency,
                self._discard_warm_up_data,
                initial_parameters,
            )

            lambda_ids = list(range(len(self._lambda_values)))
            results = pool.map(run_with_args, zip(self._lambda_values, lambda_ids))

            integral = 0.0
            variance = 0.0

            for index, result in enumerate(results):

                _, _, d_log_p_d_lambda = result

                average_d_lambda = numpy.mean(d_log_p_d_lambda)

                window_std_error = numpy.std(d_log_p_d_lambda) / numpy.sqrt(len(d_log_p_d_lambda))
                window_variance = window_std_error ** 2

                integral += average_d_lambda * self._lambda_weights[index]
                variance += self._lambda_weights[index] ** 2 * window_variance

        return results, integral, numpy.sqrt(variance)

    @staticmethod
    def _run_window(
        model,
        warm_up_steps,
        steps,
        tune_frequency,
        discard_warm_up_data,
        initial_parameters,
        lambda_tuple,
    ):

        lambda_value, lambda_index = lambda_tuple

        simulation = LambdaSimulation(
            model_collection=model,
            warm_up_steps=warm_up_steps,
            steps=steps,
            tune_frequency=tune_frequency,
            discard_warm_up_data=discard_warm_up_data,
            lambda_value=lambda_value,
        )

        trace, log_p_trace, _ = simulation.run(initial_parameters, 0, None)

        # Decorrelate the data.
        g = timeseries.statisticalInefficiency(log_p_trace, fast=False, fft=True)

        indices = timeseries.subsampleCorrelatedData(log_p_trace, g=g)

        trace = trace[indices]
        log_p_trace = log_p_trace[indices]

        print(f"Lamda Window {lambda_index}: g={g} N_samples={len(log_p_trace)}")

        # Compute d log p / d lambda
        d_lop_p_d_lambda = numpy.empty(log_p_trace.shape)

        for index in range(trace.shape[0]):
            # TODO: Vectorize this.
            d_lop_p_d_lambda[index] = model.evaluate_log_likelihood(trace[index][1:])

        return trace, log_p_trace, d_lop_p_d_lambda

"""
Code to perform 'alchemical' like 'lambda scaling free energy'
calculations to estimate Bayes factors. In these cases, lambda
is a hyperparameter which interpolates between the prior and
posterior distributions.
"""
import functools
import json
import os
from multiprocessing.pool import Pool

import numpy
from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.models import Model
from matplotlib import pyplot
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
        output_directory_path="",
        save_trace_plots=True,
        sampler=None,
        lambda_value=1.0,
    ):
        """
        Parameters
        ----------
        lambda_value: float
            The value of lambda to sample at.
        """

        super().__init__(
            model_collection,
            warm_up_steps,
            steps,
            tune_frequency,
            discard_warm_up_data,
            output_directory_path,
            save_trace_plots,
            sampler,
        )

        self._lambda = lambda_value

    def _evaluate_log_p(self, parameters, model_index):

        model = self._model_collection.models[model_index]
        return model.evaluate_log_prior(
            parameters
        ) + self._lambda * model.evaluate_log_likelihood(parameters)


class ThermodynamicIntegration:
    @property
    def lambdas(self):
        """numpy.ndarray: The location of each lambda window."""
        return self._lambda_values

    def __init__(
        self,
        legendre_gauss_degree,
        model,
        warm_up_steps=100000,
        steps=100000,
        tune_frequency=5000,
        discard_warm_up_data=True,
        output_directory_path="",
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
        output_directory_path: str
            The path to save the simulation results in.
        """

        assert isinstance(model, Model)

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

        if len(output_directory_path) > 0:
            os.makedirs(output_directory_path, exist_ok=True)

        self._output_directory_path = output_directory_path

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
                self._output_directory_path,
                initial_parameters,
            )

            lambda_ids = list(range(len(self._lambda_values)))
            results = pool.map(run_with_args, zip(self._lambda_values, lambda_ids))

            integral = 0.0
            variance = 0.0

            for index, result in enumerate(results):

                _, _, d_log_p_d_lambda = result

                average_d_lambda = numpy.mean(d_log_p_d_lambda)

                window_std_error = numpy.std(d_log_p_d_lambda) / numpy.sqrt(
                    len(d_log_p_d_lambda)
                )
                window_variance = window_std_error ** 2

                integral += average_d_lambda * self._lambda_weights[index]
                variance += self._lambda_weights[index] ** 2 * window_variance

        # Save the output
        self._save_results(results, integral, numpy.sqrt(variance))

        return results, integral, numpy.sqrt(variance)

    @staticmethod
    def _run_window(
        model,
        warm_up_steps,
        steps,
        tune_frequency,
        discard_warm_up_data,
        output_directory_path,
        initial_parameters,
        lambda_tuple,
    ):

        lambda_value, lambda_index = lambda_tuple

        lambda_directory = os.path.join(output_directory_path, str(lambda_index))

        simulation = LambdaSimulation(
            model_collection=model,
            warm_up_steps=warm_up_steps,
            steps=steps,
            tune_frequency=tune_frequency,
            discard_warm_up_data=discard_warm_up_data,
            output_directory_path=lambda_directory,
            save_trace_plots=False,
            lambda_value=lambda_value,
        )

        trace, log_p_trace, _ = simulation.run(initial_parameters, 0, None)

        # TODO: Properly decorrelate the data.
        # g = timeseries.statisticalInefficiency(log_p_trace, fast=False, fft=True)
        g = 1000.0

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

    def _save_results(self, results, integral, integral_std):
        """Saves the results of the simulation to the output
        directory.

        Parameters
        ----------
        results: tuple of tuple
            The results of each simulation in the different lambda windows
        integral: float
            The value of the integrated model evidence.
        integral_std: float
            The uncertainty in the integrated model evidence.
        """

        d_log_p_d_lambdas = numpy.zeros(len(results))
        d_log_p_d_lambdas_std = numpy.zeros(len(results))

        axis_label = r"$\dfrac{\partial \ln{p}_{\lambda}}{\partial {\lambda}}$"

        for index, result in enumerate(results):

            trace, log_p_trace, lambda_trace = result

            d_log_p_d_lambdas[index] = numpy.mean(lambda_trace)
            d_log_p_d_lambdas_std[index] = numpy.std(lambda_trace) / numpy.sqrt(
                self._steps
            )

            lambda_directory = os.path.join(self._output_directory_path, str(index))

            trace_figure = self._model.plot_trace(trace)
            trace_figure.savefig(os.path.join(lambda_directory, f"trace.pdf"))
            pyplot.close(trace_figure)

            log_p_figure = self._model.plot_log_p(lambda_trace)
            log_p_figure.savefig(os.path.join(lambda_directory, f"log_p.pdf"))
            pyplot.close(log_p_figure)

            lambda_figure = self._model.plot_log_p(lambda_trace, label=axis_label)
            lambda_figure.savefig(
                os.path.join(lambda_directory, f"d_log_p_d_lambda.pdf")
            )
            pyplot.close(lambda_figure)

        figure, axes = pyplot.subplots(1, 1, figsize=(5, 5), dpi=200)

        axes.plot(d_log_p_d_lambdas, color="#17becf")
        axes.set_xlabel(r"$\lambda$")
        axes.set_ylabel(r"$\dfrac{\partial \ln{p}_{\lambda}}{\partial {\lambda}}$")

        figure.savefig(os.path.join(self._output_directory_path, f"lambdas.pdf"))

        # Save the output as a json file and numpy files.
        results = {
            "model_evidence": integral,
            "model_evidence_std": integral_std,
            "lambdas": self._lambda_values.tolist(),
            "weights": self._lambda_weights.tolist(),
            "d_log_p_d_lambdas": d_log_p_d_lambdas.tolist(),
            "d_log_p_d_lambdas_std": d_log_p_d_lambdas_std.tolist(),
        }

        with open(
            os.path.join(self._output_directory_path, "results.json"), "w"
        ) as file:
            json.dump(results, file, sort_keys=True, indent=4, separators=(",", ": "))

        numpy.save(
            os.path.join(self._output_directory_path, "d_log_p_d_lambdas.npy"),
            d_log_p_d_lambdas,
        )
        numpy.save(
            os.path.join(self._output_directory_path, "d_log_p_d_lambdas_std.npy"),
            d_log_p_d_lambdas_std,
        )

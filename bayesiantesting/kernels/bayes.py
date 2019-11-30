"""
Code to perform 'alchemical' like 'lambda scaling free energy'
calculations to estimate Bayes factors. In these cases, lambda
is a hyperparameter which interpolates between the prior and
posterior distributions.
"""
import functools
import json
import os
import pprint
from multiprocessing.pool import Pool

import autograd
import numpy
import pymbar
from matplotlib import pyplot
from pymbar import timeseries

from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.models import Model


class LambdaSimulation(MCMCSimulation):
    """Builds an object that runs an MCMC simulation at a specific value of
    lambda - a hyperparameter which interpolates between a target distribution
    whose model evidence is intractable, and a reference distributions whose
    evidence is.

    If no reference distribution is set, the samples will be generated from the
    following simple distribution:

        p(x|D, λ) = p(x) * p(D|x) ^ λ

    or rather:

        ln p(x|D, λ) = ln p(x) + λ ln p(D|x)

    where p(x) is the prior on x, and p(D|x) is the likelihood distribution
    At λ=0.0 only the prior is sampled, at λ=1.0 the full posterior is sampled.

    If a reference model q is set, the samples will be generated according
    to:

        p(x|D, λ) = (p(x) * p(D|x)) ^ λ  * (q(x) * q(D|x)) ^ (1 - λ)

    or rather:

        ln p(x|D, λ) = λ ln p(x) + λ ln p(D|x) + (1 - λ) * q(x) + (1 - λ) * q(D|x)
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
        reference_model=None,
    ):
        """
        Parameters
        ----------
        lambda_value: float
            The value of lambda to sample at.
        reference_model: Model
            The model to transform the model of interest
            into.
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
        self._reference_model = reference_model

    def _evaluate_log_p(self, parameters, model_index):

        model = self._model_collection.models[model_index]
        return self.evaluate_log_p(
            model, parameters, self._lambda, self._reference_model
        )

    @staticmethod
    def evaluate_log_p(model, parameters, lambda_value, reference_model):
        """Evaluate the log p for a given model, set of parameters,
        and lambda value.

        Parameters
        ----------
        model: Model
            The model of interest.
        parameters: numpy.ndarray
            The parameters to evaluate the log p at with
            shape=(n_trainable_parameters).
        lambda_value: float
            The value of lambda to evaluate the log p at.
        reference_model: Model
            The model which the model of interest is being
            transformed into.

        Returns
        -------
        float
            The evaluated log p.
        """

        if reference_model is None:

            log_prior = model.evaluate_log_prior(parameters)
            log_likelihood = 0.0

            if not numpy.isclose(lambda_value, 0.0):

                log_likelihood = model.evaluate_log_likelihood(parameters)

                if not numpy.isinf(log_likelihood):
                    log_likelihood *= lambda_value

        else:

            log_prior = lambda_value * model.evaluate_log_prior(parameters) + (
                1.0 - lambda_value
            ) * reference_model.evaluate_log_prior(parameters)

            log_likelihood = lambda_value * model.evaluate_log_likelihood(
                parameters
            ) + (1.0 - lambda_value) * reference_model.evaluate_log_likelihood(
                parameters
            )

        return log_prior + log_likelihood


class BaseModelEvidenceKernel:
    """A base class for kernels which aim to estimate the
    model evidence from free energy like calculations, such
    as TI or MBAR.
    """

    @property
    def lambdas(self):
        """numpy.ndarray: The location of each lambda window."""
        return self._lambda_values

    def __init__(
        self,
        lambda_values,
        model,
        warm_up_steps=100000,
        steps=100000,
        tune_frequency=5000,
        discard_warm_up_data=True,
        output_directory_path="",
        sampler=None,
        reference_model=None,
    ):
        """
        Parameters
        ----------
        lambda_values: numpy.ndarray
            The lambda values to run simulations at, with shape=(n_lambdas).
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
        sampler: optional
            The sampler to use for in-model proposals.
        reference_model: Model
            The model to transform the model of interest
            into.
        """

        assert isinstance(model, Model)

        self._model = model
        self._warm_up_steps = warm_up_steps
        self._steps = steps
        self._tune_frequency = tune_frequency
        self._discard_warm_up_data = discard_warm_up_data
        self._sampler = sampler
        self._reference_model = reference_model

        self._lambda_values = lambda_values
        self._requires_gradients = False

        if len(output_directory_path) > 0:
            os.makedirs(output_directory_path, exist_ok=True)

        self._output_directory_path = output_directory_path

    def _validate_parameter_shapes(self, initial_parameters):
        """Validates that the initial parameters are the correct
        shape.

        Parameters
        ----------
        initial_parameters: numpy.ndarray
            The initial parameters to use in the lambda
            simulations, with shape=(n_trainable_parameters).
        """

        if len(initial_parameters) != self._model.n_trainable_parameters:

            raise ValueError(
                f"The initial parameters vector should contain "
                f"one value for each of the trainable model parameters."
            )

    def run(self, initial_parameters, number_of_processes=1):
        """Run the simulation loop.

        Parameters
        ----------
        initial_parameters: numpy.ndarray
            The initial parameters to use in the lambda
            simulations, with shape=(n_trainable_parameters).
        number_of_processes: int
            The number of processes to distribute the calculation
            across.

        Returns
        -------
        tuple of tuple
            The results of each window in the form of a tuple
            of numpy arrays (trace, log_p_trace, d_log_p_d_lamda).
        float
            The integrated model evidence.
        float
            The standard error in the model evidence.
        """

        # Make sure the parameters are the correct shape for the
        # specified model.
        self._validate_parameter_shapes(initial_parameters)

        # Simulate in each lambda window.
        with Pool(number_of_processes) as pool:

            run_with_args = functools.partial(
                BaseModelEvidenceKernel._run_window,
                self._model,
                self._warm_up_steps,
                self._steps,
                self._tune_frequency,
                self._discard_warm_up_data,
                self._output_directory_path,
                self._sampler,
                self._reference_model,
                initial_parameters,
                self._requires_gradients,
            )

            lambda_ids = list(range(len(self._lambda_values)))
            results = pool.map(run_with_args, zip(self._lambda_values, lambda_ids))

        integral, error = self._compute_integral(results)

        # Save the output
        self._save_results(results, integral, error)
        return results, integral, error

    @staticmethod
    def _run_window(
        model,
        warm_up_steps,
        steps,
        tune_frequency,
        discard_warm_up_data,
        output_directory_path,
        sampler,
        reference_model,
        initial_parameters,
        requires_gradients,
        lambda_tuple,
    ):
        """Run a given lambda window.

        Parameters
        ----------
        model: Model
            The model to sample.
        warm_up_steps: int
            The number of warm-up steps to take. During this time all
            move proposals will be tuned.
        steps: int
            The number of steps which the simulation should run for.
        tune_frequency: int
            The frequency with which to tune the move proposals.
        discard_warm_up_data: bool
            If true, all data generated during the warm-up period will
            be discarded.
        output_directory_path: str
            The path to save the simulation results in.
        sampler: optional
            The sampler to use for in-model proposals.
        reference_model: Model
            The model to transform the model of interest
            into.
        initial_parameters: numpy.ndarray
            The initial parameters to start the simulation
            from with shape=(n_trainable_parameters).
        lambda_tuple: tuple of float and int
            A tuple containing the value of lamda to simulate at,
            and the index associated with this lamda state.

        Returns
        -------
        numpy.ndarray
            The parameter trace with shape=(nsteps, n_trainable_parameters).
        numpy.ndarray
            The lop p trace with shape=(nsteps).
        numpy.ndarray
            The d lop p / d lamda trace with shape=(nsteps).
        """
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
            sampler=sampler,
            reference_model=reference_model,
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

        d_lop_p_d_lambda = numpy.empty(len(trace))

        if requires_gradients:

            # Compute d log p / d lambda
            gradient_function = autograd.grad(LambdaSimulation.evaluate_log_p, 2)

            # TODO: Vectorize this.
            for index in range(len(trace)):
                d_lop_p_d_lambda[index] = gradient_function(
                    model, trace[index][1:], lambda_value, reference_model
                )

        return trace, log_p_trace, d_lop_p_d_lambda

    def _compute_integral(self, window_results):
        """Compute the integral over all lambda windows.

        Parameters
        ----------
        window_results of tuple:
            The results of each window in the form of a tuple
            of numpy arrays (trace, log_p_trace, d_log_p_d_lamda).

        Returns
        -------
        float
            The value of the integral.
        float
            The standard error in the integral.
        """
        raise NotImplementedError()

    def _get_results_dictionary(
        self, integral, standard_error, d_log_p_d_lambdas, d_log_p_d_lambdas_std
    ):
        """Returns a dictionary containing key information about
        the results.

        Returns
        -------
        dict of str, Any
            The dictionary containing the output of this
            kernel.
        """
        return {
            "model_evidence": integral,
            "model_evidence_std": standard_error,
            "lambdas": self._lambda_values.tolist(),
            "d_log_p_d_lambdas": d_log_p_d_lambdas.tolist(),
            "d_log_p_d_lambdas_std": d_log_p_d_lambdas_std.tolist(),
        }

    def _save_results(self, results, integral, standard_error):
        """Saves the results of the simulation to the output
        directory.

        Parameters
        ----------
        results: tuple of tuple
            The results of each simulation in the different lambda windows
        integral: float
            The value of the integrated model evidence.
        standard_error: float
            The standard error in the integrated model evidence.
        """

        lambdas = numpy.zeros(len(results))
        lambdas_std = numpy.zeros(len(results))

        d_log_p_d_lambdas = numpy.zeros(len(results))
        d_log_p_d_lambdas_std = numpy.zeros(len(results))

        axis_label = r"$\dfrac{\partial \ln{p}_{\lambda}}{\partial {\lambda}}$"

        for index, result in enumerate(results):

            trace, log_p_trace, d_log_p_d_lambda = result

            d_log_p_d_lambdas[index] = numpy.mean(d_log_p_d_lambda)
            d_log_p_d_lambdas_std[index] = numpy.std(d_log_p_d_lambda) / numpy.sqrt(
                self._steps
            )

            lambdas[index] = numpy.mean(log_p_trace)
            lambdas_std[index] = numpy.std(log_p_trace) / numpy.sqrt(self._steps)

            lambda_directory = os.path.join(self._output_directory_path, str(index))

            trace_figure = self._model.plot_trace(trace)
            trace_figure.savefig(os.path.join(lambda_directory, f"trace.pdf"))
            pyplot.close(trace_figure)

            log_p_figure = self._model.plot_log_p(log_p_trace)
            log_p_figure.savefig(os.path.join(lambda_directory, f"log_p.pdf"))
            pyplot.close(log_p_figure)

            lambda_figure = self._model.plot_log_p(d_log_p_d_lambda, label=axis_label)
            lambda_figure.savefig(
                os.path.join(lambda_directory, f"d_log_p_d_lambda.pdf")
            )
            pyplot.close(lambda_figure)

        # Plot log p vs lambda
        figure, axes = pyplot.subplots(1, 1, figsize=(5, 5), dpi=200)

        axes.errorbar(self._lambda_values, lambdas, yerr=lambdas_std, color="#17becf")
        axes.set_xlabel(r"$\lambda$")
        axes.set_ylabel(r"$\ln{p}$")

        figure.savefig(
            os.path.join(self._output_directory_path, f"log_p_vs_lambda.pdf")
        )
        pyplot.close(figure)

        # Plot d log p d lambda
        figure, axes = pyplot.subplots(1, 1, figsize=(5, 5), dpi=200)

        axes.errorbar(
            self._lambda_values,
            d_log_p_d_lambdas,
            yerr=d_log_p_d_lambdas_std,
            color="#17becf",
        )
        axes.set_xlabel(r"$\lambda$")
        axes.set_ylabel(r"$\dfrac{\partial \ln{p}_{\lambda}}{\partial {\lambda}}$")

        figure.savefig(
            os.path.join(self._output_directory_path, f"d_log_p_d_lambdas.pdf")
        )
        pyplot.close(figure)

        # Save the output as a json file and numpy files.
        results = self._get_results_dictionary(
            integral, standard_error, d_log_p_d_lambdas, d_log_p_d_lambdas_std
        )

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


class ThermodynamicIntegration(BaseModelEvidenceKernel):
    """A kernel which employs thermodynamic integration and
    gaussian quadrature to estimate the model evidence.
    """

    def __init__(
        self,
        legendre_gauss_degree,
        model,
        warm_up_steps=100000,
        steps=100000,
        tune_frequency=5000,
        discard_warm_up_data=True,
        output_directory_path="",
        sampler=None,
        reference_model=None,
    ):
        """
        Parameters
        ----------
        legendre_gauss_degree: int
            The number of lambdas to use for the
            Gauss-Legendre quadrature integration.
        """

        # Choose the lambda values
        lambda_values, lambda_weights = numpy.polynomial.legendre.leggauss(
            legendre_gauss_degree
        )

        lambda_values = lambda_values * 0.5 + 0.5
        self._lambda_weights = lambda_weights * 0.5

        super().__init__(
            lambda_values,
            model,
            warm_up_steps,
            steps,
            tune_frequency,
            discard_warm_up_data,
            output_directory_path,
            sampler,
            reference_model,
        )

        self._requires_gradients = True

    def _compute_integral(self, window_results):

        integral = 0.0
        variance = 0.0

        for index, result in enumerate(window_results):

            _, _, d_log_p_d_lambda = result

            average_d_lambda = numpy.mean(d_log_p_d_lambda)

            window_std_error = numpy.std(d_log_p_d_lambda) / numpy.sqrt(
                len(d_log_p_d_lambda)
            )
            window_variance = window_std_error ** 2

            integral += average_d_lambda * self._lambda_weights[index]
            variance += self._lambda_weights[index] ** 2 * window_variance

        return integral, numpy.sqrt(variance)

    def _get_results_dictionary(
        self, integral, standard_error, d_log_p_d_lambdas, d_log_p_d_lambdas_std
    ):
        """Returns a dictionary containing key information about
        the results.

        Returns
        -------
        dict of str, Any
        """
        results = super(ThermodynamicIntegration, self)._get_results_dictionary(
            integral, standard_error, d_log_p_d_lambdas, d_log_p_d_lambdas_std
        )
        results["weights"] = self._lambda_weights.tolist()

        return results


class MBARIntegration(BaseModelEvidenceKernel):
    """A kernel which employs MBAR to estimate the model evidence.
    We define the reduced potential here as -ln p(x) - λ ln p(D|x)
    """

    def __init__(
        self,
        lambda_values,
        model,
        warm_up_steps=100000,
        steps=100000,
        tune_frequency=5000,
        discard_warm_up_data=True,
        output_directory_path="",
        sampler=None,
        reference_model=None,
    ):

        # TODO: Add trailblazing to choose these values.
        assert len(lambda_values) >= 2

        assert numpy.isclose(lambda_values[0], 0.0)
        assert numpy.isclose(lambda_values[-1], 1.0)

        super().__init__(
            lambda_values,
            model,
            warm_up_steps,
            steps,
            tune_frequency,
            discard_warm_up_data,
            output_directory_path,
            sampler,
            reference_model,
        )

        self._overlap_matrix = None

    def _compute_integral(self, window_results):

        full_trace = []
        frame_counts = numpy.empty(len(window_results))

        for index, result in enumerate(window_results):

            trace, _, _ = result
            full_trace.append(trace)

            frame_counts[index] = len(trace)

        full_trace = numpy.vstack(full_trace)

        reduced_potentials = numpy.empty((len(self._lambda_values), len(full_trace)))

        for lambda_index, lambda_value in enumerate(self._lambda_values):

            # TODO: Vectorize this.
            for trace_index in range(len(full_trace)):

                reduced_potentials[
                    lambda_index, trace_index
                ] = -LambdaSimulation.evaluate_log_p(
                    self._model,
                    full_trace[trace_index][1:],
                    lambda_value,
                    self._reference_model,
                )

        mbar = pymbar.MBAR(reduced_potentials, frame_counts)
        result = mbar.getFreeEnergyDifferences()

        self._overlap_matrix = mbar.computeOverlap()["matrix"]
        print("==============================")
        print(f"Overlap matrix:\n")
        pprint.pprint(self._overlap_matrix)
        print("==============================")

        return result["Delta_f"][-1, 0], result["dDelta_f"][-1, 0]

    def _get_results_dictionary(
        self, integral, standard_error, d_log_p_d_lambdas, d_log_p_d_lambdas_std
    ):
        """Returns a dictionary containing key information about
        the results.

        Returns
        -------
        dict of str, Any
        """
        results = super(MBARIntegration, self)._get_results_dictionary(
            integral, standard_error, d_log_p_d_lambdas, d_log_p_d_lambdas_std
        )
        results["overlap"] = self._overlap_matrix.tolist()

        return results

    @staticmethod
    def plot_overlap_matrix(overlap_matrix):
        """Plots the probability of observing a sample from state i (row)
        in state j (column). For convenience, the neigboring state cells
        are fringed in bold.

        Notes
        -----
        Modified from the https://github.com/MobleyLab/alchemical-analysis/ repository.
        """

        if not isinstance(overlap_matrix, numpy.ndarray):
            overlap_matrix = numpy.asarray(overlap_matrix)

        max_probability = numpy.max(overlap_matrix)
        n_states = len(overlap_matrix)

        figure = pyplot.figure(figsize=(n_states / 2.0, n_states / 2.0))
        figure.add_subplot(111, frameon=False, xticks=[], yticks=[])

        j = 0

        for i in range(n_states):

            if i != 0:

                pyplot.axvline(x=i, ls="-", lw=0.5, color="k", alpha=0.25)
                pyplot.axhline(y=i, ls="-", lw=0.5, color="k", alpha=0.25)

            for j in range(n_states):

                if overlap_matrix[j, i] < 0.005:
                    ii = ""
                else:
                    ii = ("%.2f" % overlap_matrix[j, i])[1:]

                alpha = overlap_matrix[j, i] / max_probability

                pyplot.fill_between(
                    [i, i + 1],
                    [n_states - j, n_states - j],
                    [n_states - (j + 1), n_states - (j + 1)],
                    color="k",
                    alpha=alpha,
                )
                pyplot.annotate(
                    ii,
                    xy=(i, j),
                    xytext=(i + 0.5, n_states - (j + 0.5)),
                    size=8,
                    textcoords="data",
                    va="center",
                    ha="center",
                    color=("k" if alpha < 0.5 else "w"),
                )

        for i in range(n_states):

            pyplot.annotate(
                i,
                xy=(i + 0.5, 1),
                xytext=(i + 0.5, n_states + 0.5),
                size=10,
                textcoords=("data", "data"),
                va="center",
                ha="center",
                color="k",
            )
            pyplot.annotate(
                i,
                xy=(-0.5, n_states - (j + 0.5)),
                xytext=(-0.5, n_states - (i + 0.5)),
                size=10,
                textcoords=("data", "data"),
                va="center",
                ha="center",
                color="k",
            )

        pyplot.annotate(
            r"$\lambda$",
            xy=(-0.5, n_states - (j + 0.5)),
            xytext=(-0.5, n_states + 0.5),
            size=10,
            textcoords=("data", "data"),
            va="center",
            ha="center",
            color="k",
        )

        pyplot.plot([0, n_states], [0, 0], "k-", lw=4.0, solid_capstyle="butt")
        pyplot.plot(
            [n_states, n_states], [0, n_states], "k-", lw=4.0, solid_capstyle="butt"
        )
        pyplot.plot([0, 0], [0, n_states], "k-", lw=2.0, solid_capstyle="butt")
        pyplot.plot(
            [0, n_states], [n_states, n_states], "k-", lw=2.0, solid_capstyle="butt"
        )

        cx = sorted(2 * list(range(n_states + 1)))
        cy = sorted(2 * list(range(n_states + 1)), reverse=True)

        pyplot.plot(cx[2:-1], cy[1:-2], "k-", lw=2.0)
        pyplot.plot(numpy.array(cx[2:-3]) + 1, cy[1:-4], "k-", lw=2.0)
        pyplot.plot(cx[1:-2], numpy.array(cy[:-3]) - 1, "k-", lw=2.0)
        pyplot.plot(cx[1:-4], numpy.array(cy[:-5]) - 2, "k-", lw=2.0)

        pyplot.xlim(-1, n_states)
        pyplot.ylim(0, n_states + 1)

        return figure

    def _save_results(self, results, integral, standard_error):

        figure = self.plot_overlap_matrix(self._overlap_matrix)
        figure.savefig(os.path.join(self._output_directory_path, f"overlap_matrix.pdf"))

        super(MBARIntegration, self)._save_results(results, integral, standard_error)

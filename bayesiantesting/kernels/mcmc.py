"""
Code to perform MCMC simulations on simple toy models.
This code was originally authored by Owen Madin (github name ocmadin).
"""
import json
import os

import numpy as np
import scipy.stats.distributions as dist
import torch

from matplotlib import pyplot
from tqdm import tqdm


from bayesiantesting.models import Model, ModelCollection
from bayesiantesting.samplers import MetropolisSampler


class MCMCSimulation:
    """ Builds an object that samples the posterior of
    a specified model.
    """

    @property
    def trace(self):
        """numpy.ndarray: A trajectory of the model parameters over the course
        of the simulation with shape=(n_steps, n_trainable_parameters+1), where
        the 'first parameter is the model index'."""
        return np.asarray(self._trace)

    @property
    def log_p_trace(self):
        """numpy.ndarray: A trajectory of the value of log p over the course
        of the simulation with shape=(n_steps,)."""
        return np.asarray(self._log_p_trace)

    @property
    def percentage_deviation_trace(self):
        """dict of str and numpy.ndarray: A trajectory of deviations of
        the models properties from their targets with shape=(n_steps)."""
        return {
            label: np.asarray(trace)
            for label, trace in self._percent_deviation_trace.items()
        }

    def __init__(
        self,
        model_collection,
        initial_parameters,
        initial_model_index=0,
        sampler=None,
        random_seed=None,
    ):
        """Initializes the basic state of the simulator object.

        Parameters
        ----------
        model_collection: Model or ModelCollection
            The model whose posterior should be sampled.
        initial_parameters: numpy.ndarray
            The initial parameters to seed the simulation with.
        initial_model_index: int
            The index of the model to begin the simulation in.
        sampler: Sampler, optional
            The sampler to use for in-model proposals. If None,
            a default `MetropolisSampler` will be used.
        random_seed: int, optional
            The random seed to use.
        """

        if isinstance(model_collection, Model):
            # Convert any standalone models to model collections
            # for convenience.
            model_collection = ModelCollection(
                model_collection.name, (model_collection,)
            )

        self._model_collection = model_collection

        # Make sure the parameters are the correct shape for the
        # specified model.
        self._validate_parameter_shapes(initial_parameters, initial_model_index)

        self._initial_values = initial_parameters
        self._initial_model_index = initial_model_index

        initial_model = self._model_collection.models[self._initial_model_index]

        # Make sure we have a sampler set
        if sampler is None:

            proposal_sizes = np.array([self._initial_values / 100]).repeat(
                self._model_collection.n_models, axis=0
            )

            proposal_sizes = np.where(proposal_sizes <= 0.0, 0.01, proposal_sizes)

            sampler = MetropolisSampler(
                self._evaluate_log_p, self._model_collection, proposal_sizes,
            )

        sampler.log_p_function = self._evaluate_log_p
        self._sampler = sampler

        # Set a random seed
        if random_seed is None:
            random_seed = torch.randint(1000000, (1,)).item()

        self._random_seed = random_seed

        torch.manual_seed(self._random_seed)
        np.random.seed(self._random_seed)

        # Set up any bookkeeping arrays
        self._has_stepped = False

        self._move_proposals = np.zeros(
            (self._model_collection.n_models, self._model_collection.n_models)
        )
        self._move_acceptances = np.zeros(
            (self._model_collection.n_models, self._model_collection.n_models)
        )

        # Set up the trace arrays.
        self._trace = []
        self._log_p_trace = []

        deviations = initial_model.compute_percentage_deviations(self._initial_values)
        self._percent_deviation_trace = {label: [] for label in deviations}

    def _validate_parameter_shapes(self, initial_parameters, initial_model_index):
        if (
            initial_model_index < 0
            or initial_model_index >= self._model_collection.n_models
        ):

            raise ValueError(
                f"The model index was outside the allowed range "
                f"[0, {self._model_collection.n_models})"
            )

        initial_log_p = self._evaluate_log_p(initial_parameters, initial_model_index)

        if np.isnan(initial_log_p) or np.isinf(initial_log_p):
            raise ValueError(f"The initial log p is NaN / inf - {initial_log_p} - initial parameters are {initial_parameters} ")

    def propagate(self, steps, warm_up=False, progress_bar=True):
        """Propagate the simulation forward by the specified number of
        `steps`. If these are flagged as `warm_up` steps, all data generated
        will be discarded and the in-model sampler will attempt to tune itself.

        Parameters
        ----------
        steps: int
            The number of steps to take.
        warm_up: bool
            Whether the treat these steps as 'warm-up' or
            'equilibration' steps.
        progress_bar: bool or tqdm.tqdm
            If False, no progress bar is printed to the terminal. If True,
            a default progress bar is printed to the terminal. If an existing
            `tqdm` progress bar, this will be used instead if the default.

        Returns
        -------
        numpy.ndarray
            The final model parameters.
        int
            The final model index.
        float
            The final value of log p
        """

        if not warm_up:
            self._has_stepped = True

        # Make sure we don't equilibrate after having already performed
        # some production steps.
        if self._has_stepped and warm_up:

            raise ValueError("The warm-up phase must come before the production phase.")

        if progress_bar is True:
            progress_bar = tqdm(total=steps + 1)

        # Initialize the starting values.
        current_parameters = self._initial_values
        current_model_index = self._initial_model_index

        current_log_p = self._evaluate_log_p(current_parameters, current_model_index)
        current_percent_deviation = self._model_collection.models[
            current_model_index
        ].compute_percentage_deviations(current_parameters)

        n_total_parameters = max(
            model.n_trainable_parameters for model in self._model_collection.models
        )

        for i in range(steps):

            # Propagate the simulation one step forward.
            (
                current_parameters,
                current_model_index,
                current_log_p,
                acceptance,
            ) = self._step(
                current_parameters, current_model_index, current_log_p, warm_up,
            )

            # Update the bookkeeping.
            if not warm_up:

                current_model = self._model_collection.models[current_model_index]

                if acceptance:

                    current_percent_deviation = current_model.compute_percentage_deviations(
                        current_parameters
                    )

                trace_padding = [0.0] * (
                    n_total_parameters - current_model.n_trainable_parameters
                )

                self._trace.append(
                    (current_model_index, *current_parameters, *trace_padding)
                )
                self._log_p_trace.append(current_log_p)

                for label in current_percent_deviation:

                    self._percent_deviation_trace[label].append(
                        current_percent_deviation[label]
                    )

            if progress_bar is not None and progress_bar is not False:
                progress_bar.update()

        if warm_up:

            # Reset the counters.
            self._move_proposals = np.zeros(
                (self._model_collection.n_models, self._model_collection.n_models)
            )
            self._move_acceptances = np.zeros(
                (self._model_collection.n_models, self._model_collection.n_models)
            )

            self._sampler.reset_counters()

        self._initial_values = current_parameters
        self._initial_model_index = current_model_index

        return current_parameters, current_model_index, current_log_p

    def _step(
        self, current_parameters, current_model_index, current_log_p, adapt_moves=False,
    ):
        """Propagates the simulation forward a single step.

        Parameters
        ----------
        current_parameters: numpy.ndarray
            The current model parameters.
        current_model_index: int
            The current model index.
        current_log_p: float
            The current value of log p.
        adapt_moves: bool
            If True, the in-model sampler will be allowed to tune itself.

        Returns
        -------
        numpy.ndarray
            The new model parameters.
        int
            The new model index.
        float
            The new value of log p.
        bool
            Whether this move was accepted or not.
        """

        proposed_parameters, proposed_log_p, acceptance = self._sampler.step(
            current_parameters, current_model_index, current_log_p, adapt_moves
        )

        self._move_proposals[current_model_index, current_model_index] += 1

        if acceptance:
            self._move_acceptances[current_model_index, current_model_index] += 1

        return proposed_parameters, current_model_index, proposed_log_p, acceptance

    def _evaluate_log_p(self, parameters, model_index):
        """Evaluates the (possibly un-normalized) target distribution
        for the given set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The parameters to evaluate at.
        model_index:
            The index of the model to evaluate.

        Returns
        -------
        float
            The evaluated log p (x).
        """
        model = self._model_collection.models[model_index]
        return model.evaluate_log_posterior(parameters)

    def print_statistics(self):
        """Print statistics about this simulation to the terminal.
        """

        print("==============================")
        print("Proposed moves:")
        print(np.sum(self._move_proposals))
        print(self._move_proposals)
        print("==============================\n")

        print("==============================")
        print("Successful moves:")
        print(self._move_acceptances)
        print("==============================\n")

        prob_matrix = self._move_acceptances / self._move_proposals

        print("==============================")
        print("Ratio of successful moves")
        print(prob_matrix)
        print("==============================\n")

        transition_matrix = np.ones(
            (self._model_collection.n_models, self._model_collection.n_models)
        )

        for i in range(self._model_collection.n_models):

            for j in range(self._model_collection.n_models):

                if i == j:
                    continue

                transition_matrix[i, i] -= transition_matrix[i, j]
                transition_matrix[i, j] = (
                    self._move_acceptances[i, j] / np.sum(self._move_proposals, 1)[i]
                )

        print("==============================")
        print("Transition matrix:")
        print(transition_matrix)
        print("==============================\n")

    def save_results(self, directory_path="", save_trace_plots=True):
        """Saves the results of this simulation to disk.

        Returns
        -------
        directory_path: str
            The directory to save the results into.
        save_trace_plots: bool
            If True, plots of the various traces will be
            generated and saved as `.pdf` files.
        """

        # Make sure the output directory exists
        if len(directory_path) > 0:
            os.makedirs(directory_path, exist_ok=True)

        # Save the traces
        self._save_traces(directory_path, save_trace_plots)

        # Save the move statistics
        self._save_statistics(directory_path)

    def _save_statistics(self, directory_path):
        """Save statistics about the simulation.

        Parameters
        ----------
        directory_path: str
            The directory to save the results into.
        """
        results = {
            "random_seed": self._random_seed,
            "move_proposals": self._move_proposals.tolist(),
            "move_acceptances": self._move_acceptances.tolist(),
            "sampler_statistics": self._sampler.get_statistics_dictionary(),
        }

        filename = os.path.join(directory_path, "statistics.json")

        with open(filename, "w") as file:
            json.dump(results, file, sort_keys=True, indent=4, separators=(",", ": "))

    def _save_traces(self, directory_path, save_trace_plots=True):
        """Saves the raw traces, as well as plots of the traces
        to the output directory.

        Parameters
        ----------
        directory_path: str
            The directory to save the results into.
        save_trace_plots: bool
            If True, plots of the various traces will be
            generated and saved as `.pdf` files.
        """

        model_counts = np.zeros(self._model_collection.n_models)

        trace = self.trace
        log_p_trace = self.log_p_trace
        percentage_deviations = self.percentage_deviation_trace

        for index, model in enumerate(self._model_collection.models):

            model_directory = os.path.join(directory_path, model.name)
            os.makedirs(model_directory, exist_ok=True)

            model_trace_indices = trace[:, 0] == index

            if not any(model_trace_indices):
                continue

            model_trace = trace[model_trace_indices]
            log_prior = []
            for counter in range(len(model_trace)):
                log_prior.append(model.evaluate_log_prior(trace[counter,1:]))
            log_prior = np.asarray(log_prior)



            model_log_p = [np.asarray(log_p_trace[model_trace_indices]),np.asarray(log_prior)]
            model_counts[index] = len(model_trace)
            model_deviations = {}

            for key in percentage_deviations:
                model_deviations[key] = percentage_deviations[key][model_trace_indices]

            if save_trace_plots:

                figures = model.plot(model_trace, model_log_p, model_deviations)

                for figure_index, file_name in enumerate(
                    ["trace.pdf", "corner.pdf", "log_p.pdf", "percentages.pdf"]
                ):
                    figures[figure_index].savefig(
                        os.path.join(model_directory, file_name)
                    )
                    pyplot.close(figures[figure_index])

            np.save(os.path.join(model_directory, "trace.npy"), model_trace[::100])
            np.save(os.path.join(model_directory, "log_p.npy"), model_log_p[0][::100])
            np.save(os.path.join(model_directory, "percentages.npy"), model_deviations[::100])

        if save_trace_plots and self._model_collection.n_models > 1:

            figure, axes = pyplot.subplots(1, 2, figsize=(10, 5))

            axes[0].plot(trace[:, 0])
            axes[1].hist(trace[:, 0])

            axes[0].set_xlabel("Model Index")
            axes[1].set_xlabel("Model Index")

            figure.savefig(os.path.join(directory_path, "model_histogram.pdf"))
            pyplot.close(figure)

    def run(
        self,
        warm_up_steps,
        steps,
        progress_bar=True,
        output_directory="",
        save_trace_plots=True,
    ):
        """A convenience function to run a production simulation
        after an initial warm-up simulation, and save the output to
        a given directory.

        Parameters
        ----------
        warm_up_steps: int
            The number of warm-up steps to take. During this time all
            move proposals will be tuned.
        steps: int
            The number of steps which the simulation should run for.
        progress_bar: bool
            If False, no progress bar is printed to the terminal. If True,
            a default progress bar is printed to the terminal.
        output_directory: str
            The path to save the simulation results in.
        save_trace_plots: bool
            If true, plots of the traces will be saved in the output
            directory.

        Returns
        -------
        numpy.ndarray
            A trajectory of the model parameters over the course
            of the simulation with shape=(n_steps, n_trainable_parameters+1),
            where the 'first parameter is the model index'.
        numpy.ndarray:
            A trajectory of the value of log p over the course
            of the simulation with shape=(n_steps,).
        dict of str and numpy.ndarray
        A trajectory of deviations of the models properties from
        their targets with shape=(n_steps).
        """

        print("==============================")
        print("Warm-up simulation:")
        self.propagate(warm_up_steps, True, progress_bar)
        print("==============================")

        print("==============================")
        print("Production simulation:")
        self.propagate(steps, False, progress_bar)
        print("==============================")

        self.print_statistics()
        self.save_results(output_directory, save_trace_plots)

        return self.trace, self.log_p_trace, self.percentage_deviation_trace

    def fit_prior_exponential(self,
    ):
        """Utility to generate exponential prior distributions
        based off the trace of a short MCMC simulation
        Inputs
        ------
        self.trace: numpy.ndarray
            A trajectory from a simulation object.

        Outputs
        -------
        param_dict: dict
        A dictionary of parameters required for the

        """
        priors = {}
        expon = dist.expon
        norm = dist.norm
        variables = ['epsilon', 'sigma', 'L', 'Q']
        for i in range(1, len(self.trace[0])):
            if i == 4:
                counts, bins = np.histogram(self.trace[:, i], range=(0, max(self.trace[:, i])))
                quadavg = np.mean(self.trace[:, i])
                for j in range(len(bins)):
                    if bins[j] < quadavg < bins[j + 1]:
                        argloc = j
                if counts[0] > 0.75*counts[argloc]:
                    prior_type = 'exponential'
                    loc = np.float64(0)
                    scale = np.mean(self.trace[:, i])
                else:
                    prior_type = 'gamma'
                    loc = 1/np.std(self.trace[:, i])
                    scale = (np.mean(self.trace[:, i])/loc)

            else:
                loc, scale = norm.fit(self.trace[:, i])
                prior_type = 'normal'
            priors[variables[i-1]] = [prior_type, [loc, scale]]

        return priors





"""
Code to perform MCMC simulations on simple toy models.
This code was originally authored by Owen Madin (github name ocmadin).
"""
import json
import os

import numpy as np
from matplotlib import pyplot
from tqdm import tqdm

from bayesiantesting.kernels.samplers import MetropolisSampler
from bayesiantesting.models import Model, ModelCollection


class MCMCSimulation:
    """ Builds an object that runs an MCMC simulation.
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
    ):
        """Initializes the basic state of the simulator object.

        Parameters
        ----------
        model_collection: Model or ModelCollection
            The model whose posterior should be sampled.
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
        save_trace_plots: bool
            If true, plots of the traces will be saved in the output
            directory.
        sampler: Sampler, optional
            The sampler to use for in-model proposals.
        """

        self.warm_up_steps = warm_up_steps
        self.steps = steps

        self._tune_frequency = tune_frequency
        self._discard_warm_up_data = discard_warm_up_data

        self._output_directory_path = output_directory_path
        self._save_trace_plots = save_trace_plots

        if isinstance(model_collection, Model):
            # Convert any standalone models to model collections
            # for convenience.
            model_collection = ModelCollection(
                model_collection.name, (model_collection,)
            )

        self._model_collection = model_collection

        self._initial_values = None
        self._initial_model_index = None
        self._initial_log_p = None

        self._sampler = sampler

        if self._sampler is not None:
            self._sampler.log_p_function = self._evaluate_log_p

    def _validate_parameter_shapes(self, initial_parameters, initial_model_index):

        if (
            initial_model_index < 0
            or initial_model_index >= self._model_collection.n_models
        ):

            raise ValueError(
                f"The model index was outside the allowed range "
                f"[0, {self._model_collection.n_models})"
            )

        maximum_n_parameters = 0

        for model in self._model_collection.models:
            maximum_n_parameters = max(
                maximum_n_parameters, model.n_trainable_parameters
            )

        # if len(initial_parameters) != maximum_n_parameters:
        #
        #     raise ValueError(
        #         f"The initial parameters vector is too small "
        #         f"({len(initial_parameters)}) to store the maximum "
        #         f"number of the trainable parameters from across "
        #         f"all models ({maximum_n_parameters})."
        #     )

    def run(self, initial_parameters, initial_model_index=0, progress_bar=True):

        # Make sure the parameters are the correct shape for the
        # specified model.
        self._validate_parameter_shapes(initial_parameters, initial_model_index)

        self._initial_values = initial_parameters
        self._initial_model_index = initial_model_index

        initial_model = self._model_collection.models[initial_model_index]

        initial_log_p = self._evaluate_log_p(self._initial_values, initial_model_index)
        initial_deviations = initial_model.compute_percentage_deviations(
            self._initial_values
        )

        if np.isnan(initial_log_p) or np.isinf(initial_log_p):
            raise ValueError(f"The initial log p is NaN / inf - {initial_log_p}")

        # Make sure we have a sampler set
        if self._sampler is None:

            self._sampler = MetropolisSampler(
                self._evaluate_log_p,
                self._model_collection,
                np.array([self._initial_values / 100]).repeat(
                    self._model_collection.n_models, axis=0
                ),
            )

        # Initialize the trace vectors
        total_steps = self.steps + self.warm_up_steps

        maximum_n_parameters = 0

        for model in self._model_collection.models:
            maximum_n_parameters = max(
                maximum_n_parameters, model.n_trainable_parameters
            )

        trace = np.zeros((total_steps + 1, maximum_n_parameters + 1))
        trace[0, 0] = self._initial_model_index
        trace[0, 1 : 1 + len(self._initial_values)] = self._initial_values

        log_p_trace = np.zeros(total_steps + 1)
        log_p_trace[0] = initial_log_p

        percent_deviation_trace = [initial_deviations]

        print(f"Markov Chain initialized values:", initial_parameters)
        print("==============================")

        print("Initial log posterior:", log_p_trace[-1])
        print("==============================")

        move_proposals = np.zeros(
            (self._model_collection.n_models, self._model_collection.n_models)
        )
        move_acceptances = np.zeros(
            (self._model_collection.n_models, self._model_collection.n_models)
        )

        print("Running Simulation...")
        print("==============================")

        if progress_bar is True:
            progress_bar = tqdm(total=self.warm_up_steps + self.steps + 1)

        for i in range(self.warm_up_steps + self.steps):

            current_model_index = int(trace[i][0])
            current_parameters = trace[i][
                1 : 1
                + self._model_collection.models[
                    current_model_index
                ].n_trainable_parameters
            ]

            current_log_p = log_p_trace[i]
            current_percent_deviation = percent_deviation_trace[i]

            # Propose the new state.
            new_parameters, new_model_index, new_log_p, acceptance = self._run_step(
                current_parameters,
                current_model_index,
                current_log_p,
                move_proposals,
                move_acceptances,
                i < self.warm_up_steps,
            )
            new_percent_deviation = current_percent_deviation

            if acceptance:

                new_percent_deviation = self._model_collection.models[
                    new_model_index
                ].compute_percentage_deviations(new_parameters)

            # Update the bookkeeping.
            trace[i + 1][0] = new_model_index
            trace[i + 1][1 : 1 + len(new_parameters)] = new_parameters

            log_p_trace[i + 1] = new_log_p
            percent_deviation_trace.append(new_percent_deviation)

            if i == self.warm_up_steps:

                move_proposals = np.zeros(
                    (self._model_collection.n_models, self._model_collection.n_models)
                )
                move_acceptances = np.zeros(
                    (self._model_collection.n_models, self._model_collection.n_models)
                )

                self._sampler.reset_counters()

            if progress_bar is not None and progress_bar is not False:
                progress_bar.update()

        if self._discard_warm_up_data:

            # Discard any warm-up data.
            trace = trace[self.warm_up_steps :]
            log_p_trace = log_p_trace[self.warm_up_steps :]

            percent_deviation_trace = percent_deviation_trace[self.warm_up_steps :]

        percent_deviation_trace_arrays = {
            label: np.zeros(len(percent_deviation_trace))
            for label in percent_deviation_trace[0]
        }

        for label in percent_deviation_trace_arrays:

            for index in range(len(percent_deviation_trace)):
                percent_deviation_trace_arrays[label][index] = percent_deviation_trace[
                    index
                ][label]

        print("Simulation Done!")
        print("==============================")

        print(f"Markov Chain final values:", trace[-1])
        print("==============================")

        self._print_statistics(move_proposals, move_acceptances)

        self._save_results(
            trace,
            log_p_trace,
            percent_deviation_trace_arrays,
            move_proposals,
            move_acceptances,
        )

        return trace, log_p_trace, percent_deviation_trace_arrays

    def _run_step(
        self,
        current_parameters,
        current_model_index,
        current_log_p,
        move_proposals,
        move_acceptances,
        adapt_moves=False,
    ):

        proposed_parameters, proposed_log_p, acceptance = self._sampler.step(
            current_parameters, current_model_index, current_log_p, adapt_moves
        )

        move_proposals[current_model_index, current_model_index] += 1

        if acceptance:
            move_acceptances[current_model_index, current_model_index] += 1

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

    def _print_statistics(self, move_proposals, move_acceptances):

        print("Proposed Moves:")

        print(np.sum(move_proposals))
        print(move_proposals)

        print("==============================")
        print("Successful Moves:")
        print(move_acceptances)

        print("==============================")

        prob_matrix = move_acceptances / move_proposals

        print("Ratio of successful moves")
        print(prob_matrix)
        print("==============================")

        transition_matrix = np.ones(
            (self._model_collection.n_models, self._model_collection.n_models)
        )

        for i in range(self._model_collection.n_models):

            for j in range(self._model_collection.n_models):

                if i == j:
                    continue

                transition_matrix[i, i] -= transition_matrix[i, j]
                transition_matrix[i, j] = (
                    move_acceptances[i, j] / np.sum(move_proposals, 1)[i]
                )

        print("Transition Matrix:")
        print(transition_matrix)
        print("==============================")

    def _save_results(
        self,
        trace,
        log_p_trace,
        percentage_deviations,
        move_proposals,
        move_acceptances,
    ):
        """Saves the results of the simulation to the output
        directory.

        Parameters
        ----------
        trace: numpy.ndarray
            The parameter trace with shape=(n_steps, n_trainable_parameters+1)
        log_p_trace: numpy.ndarray
            The log p trace with shape=(n_steps, 1)
        percentage_deviations: dict of str and numpy.ndarray
            The deviations, whose values are arrays with shape=(n_steps, 1)
        move_proposals: numpy.ndarray
            An array of the counts of the proposed moves.
        move_acceptances: numpy.ndarray
            An array of the counts of the accepted moves.
        """

        # Make sure the output directory exists
        if len(self._output_directory_path) > 0:
            os.makedirs(self._output_directory_path, exist_ok=True)

        # Save the traces
        self._save_traces(trace, log_p_trace, percentage_deviations)

        # Save the move statistics
        self._save_statistics(move_proposals, move_acceptances)

    def _save_statistics(self, move_proposals, move_acceptances):
        """Save statistics about the simulation.

        Parameters
        ----------
        move_proposals: numpy.ndarray
            An array of the counts of the proposed moves.
        move_acceptances: numpy.ndarray
            An array of the counts of the accepted moves.
        """
        results = {
            "Proposed Moves": move_proposals.tolist(),
            "Accepted Moves": move_acceptances.tolist(),
            "Sampler: ": self._sampler.get_statistics_dictionary(),
        }

        filename = os.path.join(self._output_directory_path, "statistics.json")

        with open(filename, "w") as file:
            json.dump(results, file, sort_keys=True, indent=4, separators=(",", ": "))

    def _save_traces(self, trace, log_p_trace, percentage_deviations):
        """Saves the raw traces, as well as plots of the traces
        to the output directory.

        Parameters
        ----------
        trace: numpy.ndarray
            The parameter trace with shape=(n_steps, n_trainable_parameters+1)
        log_p_trace: numpy.ndarray
            The log p trace with shape=(n_steps, 1)
        percentage_deviations: dict of str and numpy.ndarray
            The deviations, whose values are arrays with shape=(n_steps, 1)
        """

        model_counts = np.zeros(self._model_collection.n_models)

        for index, model in enumerate(self._model_collection.models):

            model_directory = os.path.join(self._output_directory_path, model.name)

            if len(model_directory) > 0:
                os.makedirs(model_directory, exist_ok=True)

            model_trace_indices = trace[:, 0] == index

            if not any(model_trace_indices):
                continue

            model_trace = trace[model_trace_indices]
            model_log_p = log_p_trace[model_trace_indices]
            model_counts[index] = len(model_trace)
            model_deviations = {}

            for key in percentage_deviations:
                model_deviations[key] = percentage_deviations[key][model_trace_indices]

            if self._save_trace_plots:

                figures = model.plot(model_trace, model_log_p, model_deviations)

                for figure_index, file_name in enumerate(
                    ["trace.pdf", "corner.pdf", "log_p.pdf", "percentages.pdf"]
                ):
                    figures[figure_index].savefig(
                        os.path.join(model_directory, file_name)
                    )
                    pyplot.close(figures[figure_index])

            np.save(os.path.join(model_directory, "trace.npy"), model_trace)
            np.save(os.path.join(model_directory, "log_p.npy"), model_log_p)
            np.save(os.path.join(model_directory, "percentages.npy"), model_deviations)

        if self._save_trace_plots:

            figure, axes = pyplot.subplots(1, 2, figsize=(10, 5))

            axes[0].plot(trace[:, 0])
            axes[1].hist(trace[:, 0])

            axes[0].set_xlabel("Model Index")
            axes[1].set_xlabel("Model Index")

            figure.savefig(
                os.path.join(self._output_directory_path, "model_histogram.pdf")
            )
            pyplot.close(figure)

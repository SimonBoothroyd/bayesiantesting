"""
Code to perform RJMC simulations on simple toy models.
This code was originally authored by Owen Madin (github name ocmadin).
"""
import math

import numpy as np
import torch

from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.models.models import ModelCollection


class RJMCSimulation(MCMCSimulation):
    """ Builds an object that runs an RJMC simulation based
    on the parameters the user gives to it
    """

    def __init__(
        self,
        model_collection,
        initial_parameters,
        initial_model_index=0,
        sampler=None,
        random_seed=None,
        swap_frequency=0.3,
    ):
        """
        Parameters
        ----------
        swap_frequency: float
            The percentage of times the simulation tries to jump between models.
        """
        if not isinstance(model_collection, ModelCollection):
            raise ValueError("The model must be a `ModelCollection`.")

        if not model_collection.n_models > 1:

            raise ValueError(
                "The model collection must contain at least two "
                "sub-models to jump between."
            )

        super().__init__(
            model_collection,
            initial_parameters,
            initial_model_index,
            sampler,
            random_seed,
        )

        self._swap_frequency = swap_frequency

    def _step(
        self,
        current_parameters,
        current_model_index,
        current_log_p,
        adapt=False,
    ):

        proposed_parameters = current_parameters.copy()
        proposed_model_index = int(current_model_index)

        random_move = torch.rand((1,)).item()

        if random_move <= self._swap_frequency:

            # Propose a cross-model move.
            (
                proposed_parameters,
                proposed_log_p,
                proposed_model_index,
                jacobian,
                transition_probability,
            ) = self.model_proposal(proposed_parameters, proposed_model_index)

            alpha = (
                (proposed_log_p - current_log_p)
                + np.log(jacobian)
                + np.log(transition_probability)
            )

            # Check for NaNs in the proposed state.
            if proposed_log_p == math.nan:

                alpha = -math.inf
                proposed_log_p = -math.inf

            # Apply the acceptance criteria.
            random_number = torch.rand((1,)).item()
            acceptance = np.log(random_number) < alpha

        else:

            # Propose an in-model move.
            proposed_parameters, proposed_log_p, acceptance = self._sampler.step(
                proposed_parameters, current_model_index, current_log_p, adapt
            )

        self._move_proposals[current_model_index, proposed_model_index] += 1

        if acceptance:

            new_log_p = proposed_log_p
            new_params = proposed_parameters
            new_model_index = proposed_model_index

            self._move_acceptances[current_model_index, new_model_index] += 1

        else:

            new_log_p = current_log_p
            new_params = current_parameters
            new_model_index = current_model_index

        return new_params, new_model_index, new_log_p, acceptance

    def model_proposal(self, current_parameters, current_model_index):

        proposed_model_index = int(current_model_index)

        # Propose new model to jump to
        while proposed_model_index == current_model_index:
            proposed_model_index = torch.randint(
                self._model_collection.n_models, (1,)
            ).item()

        _, proposed_parameters, jacobian_array = self._model_collection.map_parameters(
            current_parameters, current_model_index, proposed_model_index
        )

        proposed_log_p = self._evaluate_log_p(proposed_parameters, proposed_model_index)

        jacobian = np.prod(jacobian_array)
        transition_probability = self._model_collection.transition_probabilities(
            current_model_index, proposed_model_index
        )

        # Return values of jacobian in order to properly calculate accept/reject
        return (
            proposed_parameters,
            proposed_log_p,
            proposed_model_index,
            jacobian,
            transition_probability,
        )


class BiasedRJMCSimulation(RJMCSimulation):
    """An extension of the `RJMCSimulation` class which allows the
    user to specify biases on specific models.
    """

    def __init__(
        self,
        model_collection,
        initial_parameters,
        initial_model_index=0,
        sampler=None,
        random_seed=None,
        swap_frequency=0.3,
        log_biases=None,
    ):
        """
        Parameters
        ----------
        log_biases: numpy.ndarray
            The log biasing factors to add to the posterior
            distribution of each model (shape=(model_collection.n_models)).
        """
        super().__init__(
            model_collection,
            initial_parameters,
            initial_model_index,
            sampler,
            random_seed,
            swap_frequency,
        )

        self._log_biases = log_biases
        assert len(log_biases) == model_collection.n_models

    def _evaluate_log_p(self, parameters, model_index):
        return (
            super(BiasedRJMCSimulation, self)._evaluate_log_p(parameters, model_index)
            + self._log_biases[model_index]
        )

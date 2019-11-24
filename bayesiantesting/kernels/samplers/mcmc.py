"""
This module implements a samplers based off of the
Metropolis-Hasting acceptance criteria. These samplers
do not require nor make use of gradient information.
"""
import numpy
import torch

from bayesiantesting.kernels.samplers.samplers import Sampler
from bayesiantesting.utils import distributions


class MetropolisSampler(Sampler):
    """A base class for different in-model parameter samplers."""

    @property
    def proposal_sizes(self):
        """numpy.ndarray: The size of the proposals to make for each
        parameter with shape=(n_models, n_parameters).
        """
        return self._proposal_sizes

    @proposal_sizes.setter
    def proposal_sizes(self, value):

        assert value.shape == (self._model_collection.n_models, self._max_n_parameters,)
        self._proposal_sizes = value

    def __init__(
        self,
        log_p_function,
        model_collection,
        proposal_sizes,
        acceptance_target=0.5,
        tune_frequency=100,
    ):
        """Initializes self.

        Parameters
        ----------
        proposal_sizes: numpy.ndarray, optional
            The size of the proposals to make for each parameter
            with shape=(n_models, n_parameters).
        acceptance_target: float
            The target acceptance rate for this sampler
        tune_frequency: int
            The number of steps to take before attempting to
            tune the parameters.
        """
        super().__init__(log_p_function, model_collection)

        assert 0.0 < acceptance_target <= 1.0
        self._acceptance_target = acceptance_target

        self.proposal_sizes = proposal_sizes

        self._tune_frequency = tune_frequency

    def step(self, parameters, model_index, log_p, adapt):

        model = self._model_collection.models[model_index]

        # Choose a random parameter to change
        parameter_index = torch.randint(model.n_trainable_parameters, (1,))

        # Sample the new parameters from a normal distribution.
        proposed_parameters = parameters.copy()

        proposed_parameters[parameter_index] = distributions.Normal(
            parameters[parameter_index],
            self._proposal_sizes[model_index][parameter_index],
        ).sample()

        proposed_log_p = self._log_p_function(proposed_parameters, model_index)

        alpha = proposed_log_p - log_p

        random_number = numpy.log(torch.rand((1,)).item())
        accept = random_number < alpha

        # Update the bookkeeping
        self._proposed_moves[parameter_index] += 1

        if accept:

            self._accepted_moves[parameter_index] += 1

            parameters = proposed_parameters
            log_p = proposed_log_p

        # Tune the proposals if needed
        total_proposed_moves = numpy.sum(self._proposed_moves)

        if (
            adapt
            and self._tune_frequency > 0
            and total_proposed_moves > 0
            and total_proposed_moves % self._tune_frequency == 0
        ):
            self._tune_proposals()

        return parameters, log_p, accept

    def _tune_proposals(self):
        """Attempt to tune the move proposals to reach the
        `acceptance_target`.
        """

        for index, model in enumerate(self._model_collection.models):

            divisor = numpy.maximum(1, self._proposed_moves[index])
            acceptance_rates = self._accepted_moves[index] / divisor

            for parameter_index, rate in enumerate(acceptance_rates):

                scale = 0.9 if rate < self._acceptance_target else 1.1
                scale = (
                    1.0 if self._proposed_moves[index][parameter_index] == 0 else scale
                )

                self._proposal_sizes[index][parameter_index] *= scale

        self.reset_counters()

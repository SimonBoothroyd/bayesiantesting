import abc

import autograd
import numpy

from bayesiantesting.models import ModelCollection


class Sampler(abc.ABC):
    """A base class for different in-model parameter samplers."""

    @property
    def log_p_function(self):
        """function: The log p function to sample over.
        """
        return self._log_p_function

    @log_p_function.setter
    def log_p_function(self, value):

        self._log_p_function = value
        self._gradient_function = None if value is None else autograd.grad(value)

    @property
    def proposed_moves(self):
        """numpy.ndarray: The number of moves this sampler has
        proposed for each parameter with shape=(n_parameters).
        """
        return self._proposed_moves

    @property
    def accepted_moves(self):
        """numpy.ndarray: The number of moves this sampler has
        accepted for each parameter with shape=(n_parameters).
        """
        return self._accepted_moves

    def __init__(self, log_p_function, model_collection):
        """Initializes self.

        Parameters
        ----------
        log_p_function: function, optional
            The log probability function to sample.
        model_collection: ModelCollection
            The model whose parameters are being sampled.
        """

        assert isinstance(model_collection, ModelCollection)

        self._log_p_function = None
        self._gradient_function = None

        self.log_p_function = log_p_function

        self._model_collection = model_collection

        self._max_n_parameters = max(
            (model.n_trainable_parameters for model in model_collection.models)
        )

        self._proposed_moves = numpy.zeros(
            (self._model_collection.n_models, self._max_n_parameters)
        )
        self._accepted_moves = numpy.zeros(
            (self._model_collection.n_models, self._max_n_parameters)
        )

    def reset_counters(self):
        """Resets this samplers count of the number of
        proposed and accepted moves.
        """
        self._proposed_moves = numpy.zeros(
            (self._model_collection.n_models, self._max_n_parameters)
        )
        self._accepted_moves = numpy.zeros(
            (self._model_collection.n_models, self._max_n_parameters)
        )

    @abc.abstractmethod
    def step(self, parameters, model_index, log_p, adapt):
        """Propagates a set of parameters forward one step.

        Parameters
        ----------
        parameters: numpy.ndarray
            The parameters to propagate with shape=(n_params)
        model_index: int
            The index of the model to sample in.
        log_p: float
            The value of log p evaluated at the current `parameters`.
        adapt: bool
            If True, this sampler will attempt to tune it's
            parameters for optimal sampling.

        Returns
        -------
        numpy.ndarray
            The new parameters.
        float
            The value of log p evaluated at the new parameters.
        bool
            Whether this move was accepted or not.
        """
        raise NotImplementedError()

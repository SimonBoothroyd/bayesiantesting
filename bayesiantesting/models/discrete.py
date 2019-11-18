"""
A collection of model collections i.e collections
of linked models which we wish to sample between
either with techniques such as RJMC, or to employ
free-energy like techniques to evaluate the individual
model evidences.

Models in this module should inherit from the `ModelCollection`
subclass.
"""
from bayesiantesting.models import ModelCollection
from bayesiantesting.models.continuous import TwoCenterLJModel


class TwoCenterLJModelCollection(ModelCollection):
    """A collection of two-center lennard-jones models,
    which can be transitioned between using techniques such
    as RJMC.
    """

    def __init__(self, name, models, maximum_a_posteriori=None):
        """
        Parameters
        ----------
        maximum_a_posteriori: List of numpy.ndarray, optional
            The maximum a posteriori values of the trainable
            parameters. These are used to map the parameters
            between models during RJMC model proposal moves. If
            None, the default mapping based on the model priors
            will be used.
        """

        supported_models = ["AUA", "AUA+Q", "UA"]

        assert all(isinstance(model, TwoCenterLJModel) for model in models)
        assert all(model.name in supported_models for model in models)

        super().__init__(name, models)

        self._maximum_a_posteriori = maximum_a_posteriori

        if self._maximum_a_posteriori is not None:

            assert len(self._maximum_a_posteriori) == self.n_models

            for values, model in zip(self._maximum_a_posteriori, self.models):
                assert len(values) == model.n_trainable_parameters

    def _mapping_function(
        self, parameter, model_index_a, model_index_b, parameter_index
    ):

        model_a = self._models[model_index_a]
        model_b = self._models[model_index_b]

        if (
            parameter_index < model_a.n_trainable_parameters
            and parameter_index < model_b.n_trainable_parameters
            and self._maximum_a_posteriori is not None
        ):

            ratio = (self._maximum_a_posteriori[model_index_b][parameter_index] /
                     self._maximum_a_posteriori[model_index_a][parameter_index])

            return ratio * parameter

        return super(TwoCenterLJModelCollection, self)._mapping_function(
            parameter, model_index_a, model_index_b, parameter_index
        )

    # def transition_probabilities(self):
    #
    #     unif = distributions.uniform.pdf
    #
    #     transition_matrix = np.ones((self.model.n_models, self.model.n_models))
    #
    #     # These are proposal distributions for "new" variables (that exist in one
    #     # model but not the other).  They have been cleverly chosen to all equal 1
    #     g_0_1 = unif(self.w, 0, 1)
    #     g_1_0 = 1
    #     g_0_2 = 1
    #     g_2_0 = 1
    #
    #     # These are probabilities of proposing a model from one model to another.
    #     # The probability is half for moves originating in AUA because they can
    #     # move either to UA or AUA+Q. We disallow moves between UA and AUA+Q
    #     # directly
    #     q_0_1 = 1 / 2
    #     q_1_0 = 1
    #     q_0_2 = 1 / 2
    #     q_2_0 = 1
    #
    #     # Note that this is really times swap_freq but that term always cancels.
    #     transition_matrix[0, 1] = g_1_0 * q_1_0 / (g_0_1 * q_0_1)
    #     transition_matrix[1, 0] = g_0_1 * q_0_1 / (g_1_0 * q_1_0)
    #     transition_matrix[0, 2] = g_2_0 * q_2_0 / (g_0_2 * q_0_2)
    #     transition_matrix[2, 0] = g_0_2 * q_0_2 / (g_2_0 * q_2_0)
    #
    #     return transition_matrix

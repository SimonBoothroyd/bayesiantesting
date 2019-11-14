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

    def __init__(self, name, models):

        supported_models = ["AUA", "AUA+Q", "UA"]

        assert all(isinstance(model, TwoCenterLJModel) for model in models)
        assert all(model.name in supported_models for model in models)

        super().__init__(name, models)

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
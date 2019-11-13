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

        supported_models = ['AUA', 'AUA+Q', 'UA']

        assert all(isinstance(model, TwoCenterLJModel) for model in models)
        assert all(model.name in supported_models for model in models)

        self._model_name_to_index = {model.name: index for index, model in enumerate(models)}

        self._allowed_mappings = {
            'UA': ['AUA'],
            'AUA': ['UA', 'AUA+Q'],
            'AUA+Q': ['AUA']
        }

        super().__init__(name, models)

    def transition_function(self):

        unif = distributions.uniform.pdf

        transition_matrix = np.ones((self.model.n_models, self.model.n_models))

        # These are proposal distributions for "new" variables (that exist in one
        # model but not the other).  They have been cleverly chosen to all equal 1
        g_0_1 = unif(self.w, 0, 1)
        g_1_0 = 1
        g_0_2 = 1
        g_2_0 = 1

        # These are probabilities of proposing a model from one model to another.
        # The probability is half for moves originating in AUA because they can
        # move either to UA or AUA+Q. We disallow moves between UA and AUA+Q
        # directly
        q_0_1 = 1 / 2
        q_1_0 = 1
        q_0_2 = 1 / 2
        q_2_0 = 1

        # Note that this is really times swap_freq but that term always cancels.
        transition_matrix[0, 1] = g_1_0 * q_1_0 / (g_0_1 * q_0_1)
        transition_matrix[1, 0] = g_0_1 * q_0_1 / (g_1_0 * q_1_0)
        transition_matrix[0, 2] = g_2_0 * q_2_0 / (g_0_2 * q_0_2)
        transition_matrix[2, 0] = g_0_2 * q_0_2 / (g_2_0 * q_2_0)

        return transition_matrix

    def jacobian(self):

        jacobian_matrix = np.ones((self.n_models, self.n_models))

        if self.optimum_matching == "True":
            jacobian_matrix[0, 1] = (
                (1 / (self.lamda * self.w))
                * (
                    self.opt_params_AUA_Q[0]
                    * self.opt_params_AUA_Q[1]
                    * self.opt_params_AUA_Q[2]
                )
                / (
                    self.opt_params_AUA[0]
                    * self.opt_params_AUA[1]
                    * self.opt_params_AUA[2]
                )
            )
            jacobian_matrix[1, 0] = (
                self.lamda
                * (
                    self.opt_params_AUA[0]
                    * self.opt_params_AUA[1]
                    * self.opt_params_AUA[2]
                )
                / (
                    self.opt_params_AUA_Q[0]
                    * self.opt_params_AUA_Q[1]
                    * self.opt_params_AUA_Q[2]
                )
            )
            jacobian_matrix[0, 1] = 1 / (self.lamda * self.w)
            jacobian_matrix[1, 0] = self.w * self.lamda
        else:
            jacobian_matrix[0, 1] = 1 / (self.lamda * self.w)
            jacobian_matrix[1, 0] = self.w * self.lamda

        # Optimum Matching for UA --> AUA
        # jacobian[0,1]=(1/(lamda*w))*(opt_params_AUA_Q[0]*opt_params_AUA_Q[1]*
        # opt_params_AUA_Q[2])/(opt_params_AUA[0]*opt_params_AUA[1]*opt_params_AUA[2])
        # jacobian[1,0]=lamda*(opt_params_AUA[0]*opt_params_AUA[1]*opt_params_AUA[2])
        # /(opt_params_AUA_Q[0]*opt_params_AUA_Q[1]*opt_params_AUA_Q[2])
        jacobian_matrix[0, 2] = (
            self.opt_params_UA[0] * self.opt_params_UA[1] * self.opt_params_UA[2]
        ) / (self.opt_params_AUA[0] * self.opt_params_AUA[1] * self.opt_params_AUA[2])
        jacobian_matrix[2, 0] = (
            self.opt_params_AUA[0] * self.opt_params_AUA[1] * self.opt_params_AUA[2]
        ) / (self.opt_params_UA[0] * self.opt_params_UA[1] * self.opt_params_UA[2])
        # Direct transfer for AUA->AUA+Q

        return jacobian_matrix

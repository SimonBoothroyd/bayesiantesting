import numpy as np

from bayesiantesting.models import BaseModel


class TwoCenterLennardJones(BaseModel):
    """A representation of the two-center Lennard-Jones model, which
    can be evaluated using a surrogate model against a `NISTDataSet`.
    """

    def __init__(
        self, prior_settings, reference_data_set, property_types, surrogate_model
    ):
        """Constructs a new `TwoCenterLennardJones` model.

        Parameters
        ----------
        reference_data_set: NISTDataSet
            The data set which contains the experimentally observed values
            that this model is attempts to reproduce.
        property_types: list of NISTDataType
            The list of property types which this model attempts
            to reproduce.
        surrogate_model: SurrogateModel
            The surrogate model to use when evaluating the likelihood function.
        """
        super().__init__(prior_settings)

        self._property_types = property_types

        self._reference_data = {}
        self._reference_precisions = {}

        for property_type in self._property_types:

            self._reference_data[property_type] = np.asarray(
                reference_data_set.get_data(property_type)
            )
            self._reference_precisions[property_type] = np.asarray(
                reference_data_set.get_precision(property_type)
            )

        self._surrogate_model = surrogate_model

    def evaluate_log_likelihood(self, parameters):
        """Evaluates the log value of the this models likelihood for
        a set of parameters. based on a gaussian penalty function.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=(n parameters, 1))
            to evaluate at.

        Returns
        -------
        float
            The log value of the likelihood evaluated at `parameters`.
        """
        from scipy.stats import distributions as scipy_distributions

        log_p = 0.0

        for property_type in self._property_types:

            reference_data = self._reference_data[property_type]
            precision = self._reference_precisions[property_type]

            temperatures = reference_data[:, 0]

            reference_values = reference_data[:, 1]
            surrogate_values = self._surrogate_model.evaluate(
                property_type, parameters, temperatures
            )

            # Compute likelihood based on gaussian penalty function
            log_p += sum(
                scipy_distributions.norm.logpdf(
                    reference_values, surrogate_values, precision ** -2.0
                )
            )

        return log_p

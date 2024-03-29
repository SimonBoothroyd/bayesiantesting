"""
A collection of trainable models whose parameters are
all continuous.

Models in this module should inherit from the `Model`
subclass.
"""

import autograd.numpy
import numpy

from bayesiantesting import unit
from bayesiantesting.models import Model
from bayesiantesting.utils import distributions as distributions


class UnconditionedModel(Model):
    """Represents a model which isn't conditioned upon
    any data and with a uniform likelihood - i.e. a model
    with only priors and log p (D|x) = 0.0.
    """

    def evaluate_log_likelihood(self, parameters):
        return 0.0


class TwoCenterLJModel(Model):
    """A representation of the two-center Lennard-Jones model, which
    can be evaluated using a surrogate model against a `NISTDataSet`.
    """

    def __init__(
        self,
        name,
        prior_settings,
        fixed_parameters,
        reference_data_set,
        property_types,
        surrogate_model,
    ):
        """Constructs a new `TwoCenterLJModel` model.

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
        super().__init__(name, prior_settings, fixed_parameters)

        required_parameters = ["epsilon", "sigma", "L", "Q"]
        provided_parameters = [*self._prior_labels, *self._fixed_labels]

        missing_parameters = set(required_parameters) - set(provided_parameters)
        extra_parameters = set(provided_parameters) - set(required_parameters)

        if len(missing_parameters) > 0:
            raise ValueError(
                f"The {', '.join(missing_parameters)} parameters are required but were not provided."
            )
        if len(extra_parameters) > 0:
            raise ValueError(
                f"The {', '.join(extra_parameters)} parameters are not supported by this model."
            )

        self._property_types = property_types

        self._reference_data = {}
        self._reference_precisions = {}

        self._critical_temperature = reference_data_set.critical_temperature.value
        self._critical_temperature = self._critical_temperature.to(
            unit.kelvin
        ).magnitude

        for property_type in self._property_types:

            self._reference_data[property_type] = numpy.asarray(
                reference_data_set.get_data(property_type)
            )
            self._reference_precisions[property_type] = numpy.asarray(
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
        log_p = 0.0

        all_parameters = numpy.array([*parameters, *self._fixed_parameters])

        for property_type in self._property_types:

            reference_data = self._reference_data[property_type]
            precisions = self._reference_precisions[property_type]

            temperatures = reference_data[:, 0]

            reference_values = reference_data[:, 1]
            surrogate_values = self._surrogate_model.evaluate(
                property_type, all_parameters, temperatures
            )

            surrogate_values = surrogate_values
            precisions = precisions ** -2.0
            reference_values = reference_values

            if (
                any(autograd.numpy.isnan(surrogate_values))
                or any(autograd.numpy.isinf(surrogate_values))
                or any(surrogate_values > 1e10)
            ):
                return -numpy.inf

            # Compute likelihood based on gaussian penalty function
            log_p += autograd.numpy.sum(
                distributions.Normal(surrogate_values, precisions).log_pdf(
                    reference_values
                )
            )

        return log_p

    def compute_percentage_deviations(self, parameters):

        deviations = {}

        all_parameters = numpy.array([*parameters, *self._fixed_parameters])

        for property_type in self._property_types:

            reference_data = self._reference_data[property_type]

            reference_values = reference_data[:, 1]
            surrogate_values = self._surrogate_model.evaluate(
                property_type, all_parameters, reference_data[:, 0]
            )

            deviation_vector = (
                (reference_values - surrogate_values) / reference_values
            ) ** 2
            mean_percentage_deviation = numpy.sqrt(numpy.mean(deviation_vector)) * 100

            deviations[property_type] = mean_percentage_deviation

        return deviations


class GaussianModel(Model):
    """A toy model with a gaussian likelihood function with is
    not conditioned on any data.
    """

    def __init__(self, name, prior_settings, loc, scale, weight=1.0):

        super().__init__(name, prior_settings, {})

        self._loc = loc
        self._scale = scale

        self._log_weight = numpy.log(weight)

    def evaluate_log_likelihood(self, parameters):
        return (
            distributions.Normal(self._loc, self._scale).log_pdf(parameters)
            + self._log_weight
        )

    def compute_percentage_deviations(self, parameters):
        return {}

    def to_json(self):
        raise NotImplementedError()

    @classmethod
    def from_json(cls, json_string):
        raise NotImplementedError()


class CauchyModel(Model):
    """A toy model with a cauchy likelihood function with is
    not conditioned on any data.
    """

    def __init__(self, name, prior_settings, loc, scale, weight=1.0):

        super().__init__(name, prior_settings, {})

        self._loc = loc
        self._scale = scale

        self._log_weight = numpy.log(weight)

    def evaluate_log_likelihood(self, parameters):
        return (
            distributions.Cauchy(self._loc, self._scale).log_pdf(parameters)
            + self._log_weight
        )

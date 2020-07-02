import autograd
import autograd.numpy
import numpy
import theano.tensor as tt

from bayesiantesting import unit
from bayesiantesting.utils import distributions


class StollWerthOp(tt.Op):

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, reference_data_set, property_types, surrogate_model):

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

        self._log_p_grad = autograd.grad(self._lop_p)

    def _lop_p(self, theta):

        log_p = 0.0

        for property_type in self._property_types:
            reference_data = self._reference_data[property_type]
            precisions = self._reference_precisions[property_type]

            temperatures = reference_data[:, 0]

            reference_values = reference_data[:, 1]
            surrogate_values = self._surrogate_model.evaluate(
                property_type, theta, temperatures
            )

            precisions = precisions ** -2.0

            if (
                any(numpy.isnan(surrogate_values))
                or any(numpy.isinf(surrogate_values))
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

    def perform(self, node, inputs, outputs, **kwargs):

        (theta,) = inputs
        outputs[0][0] = numpy.array(self._lop_p(theta))

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs

        if theta.ndim == 1:
            return [tt.zeros(4)]

        d_log_p_d_theta = self._log_p_grad(numpy.asarray(theta))
        return [g[0] * d_log_p_d_theta]

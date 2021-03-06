"""
A set of common distributions which are differentiable with
autograd.
"""
import abc

import autograd
import autograd.numpy
import autograd.scipy.special
import autograd.scipy.stats.gamma
import torch.distributions

from bayesiantesting.utils.serializeable import Serializable


class Distribution(Serializable):
    @property
    def n_variables(self):
        """int: The number of variables which this distribution is a
        function of."""
        return 1

    @abc.abstractmethod
    def log_pdf(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def cdf(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse_cdf(self, x):
        raise NotImplementedError()


class MultivariateDistribution(Distribution, abc.ABC):
    """A distribution which a function of more than
    one variable.
    """

    @property
    @abc.abstractmethod
    def n_variables(self):
        raise NotImplementedError()


class MultivariateNormal(MultivariateDistribution):
    """A multivariate normal distribution.
    """

    @property
    def n_variables(self):
        return self._dimension

    def __init__(self, mean, covariance):

        self._mean = mean
        self._dimension = len(self._mean)

        assert len(covariance.shape) == 2
        assert covariance.shape[0] == covariance.shape[1] == self._dimension

        self._covariance = covariance

        # noinspection PyUnresolvedReferences
        self._inverse_covariance = autograd.numpy.linalg.inv(covariance)
        # noinspection PyUnresolvedReferences
        self._log_determinant = autograd.numpy.log(
            autograd.numpy.linalg.det(covariance)
        )

    def log_pdf(self, x):

        residuals = x - self._mean

        # noinspection PyUnresolvedReferences
        log_p = -0.5 * (
            self._log_determinant
            + autograd.numpy.einsum(
                "...j,jk,...k", residuals, self._inverse_covariance, residuals
            )
            + self._dimension * autograd.numpy.log(2 * autograd.numpy.pi)
        )

        return log_p

    def cdf(self, x):
        raise NotImplementedError()

    def inverse_cdf(self, x):
        raise NotImplementedError()

    def sample(self):

        torch_mean = torch.tensor(self._mean, dtype=torch.float64)
        torch_covariance = torch.tensor(self._covariance, dtype=torch.float64)

        distribution = torch.distributions.MultivariateNormal(
            torch_mean, torch_covariance
        )
        return distribution.rsample().numpy()

    def to_dict(self):
        return {"mean": self._mean.tolist(), "covariance": self._covariance.tolist()}

    @classmethod
    def from_dict(cls, dictionary):
        super(MultivariateNormal, cls).from_dict(dictionary)
        # noinspection PyUnresolvedReferences
        return cls(
            autograd.numpy.asarray(dictionary["mean"]),
            autograd.numpy.asarray(dictionary["covariance"]),
        )

    @staticmethod
    def _validate(dictionary):
        assert "mean" in dictionary and "covariance" in dictionary


class Exponential(Distribution):
    def __init__(self, rate):
        self.rate = rate

    def log_pdf(self, x):

        if x < 0.0:
            # noinspection PyUnresolvedReferences
            return -autograd.numpy.inf

        # noinspection PyUnresolvedReferences
        return autograd.numpy.log(self.rate) - self.rate * x

    def cdf(self, x):
        # noinspection PyUnresolvedReferences
        return 1 - autograd.numpy.exp(-self.rate * x)

    def inverse_cdf(self, x):
        # noinspection PyUnresolvedReferences
        return -autograd.numpy.log(1 - x) / self.rate

    def sample(self):
        return torch.distributions.Exponential(self.rate).rsample().item()

    def to_dict(self):
        return {"rate": self.rate}

    @classmethod
    def from_dict(cls, dictionary):
        super(Exponential, cls).from_dict(dictionary)
        return cls(dictionary["rate"])

    @staticmethod
    def _validate(dictionary):
        assert "rate" in dictionary and dictionary["rate"] >= 0.0


class Normal(Distribution):
    def __init__(self, loc, scale):

        self.loc = loc
        self.scale = scale

    def log_pdf(self, x):

        var = self.scale ** 2
        # noinspection PyUnresolvedReferences
        log_scale = autograd.numpy.log(self.scale)

        # noinspection PyUnresolvedReferences
        return (
            -((x - self.loc) ** 2) / (2 * var)
            - log_scale
            - autograd.numpy.log(autograd.numpy.sqrt(2 * autograd.numpy.pi))
        )

    def cdf(self, x):
        # noinspection PyUnresolvedReferences
        return 0.5 * (
            1
            + autograd.scipy.special.erf(
                (x - self.loc) * (1.0 / self.scale) / autograd.numpy.sqrt(2)
            )
        )

    def inverse_cdf(self, x):
        # noinspection PyUnresolvedReferences
        return self.loc + self.scale * autograd.scipy.special.erfinv(
            2 * x - 1
        ) * autograd.numpy.sqrt(2)

    def sample(self):
        return torch.distributions.Normal(self.loc, self.scale).rsample().item()

    def to_dict(self):
        return {"loc": self.loc, "scale": self.scale}

    @classmethod
    def from_dict(cls, dictionary):
        super(Normal, cls).from_dict(dictionary)
        return cls(dictionary["loc"], dictionary["scale"])

    @staticmethod
    def _validate(dictionary):
        assert (
            "loc" in dictionary and "scale" in dictionary and dictionary["scale"] >= 0.0
        )


class Cauchy(Distribution):
    def __init__(self, loc, scale):

        self.loc = loc
        self.scale = scale

    def log_pdf(self, x):
        # noinspection PyUnresolvedReferences
        return (
            -autograd.numpy.log(autograd.numpy.pi)
            - autograd.numpy.log(self.scale)
            - autograd.numpy.log(1 + ((x - self.loc) / self.scale) ** 2)
        )

    def cdf(self, x):
        # noinspection PyUnresolvedReferences
        return (
            autograd.numpy.arctan((x - self.loc) / self.scale) / autograd.numpy.pi + 0.5
        )

    def inverse_cdf(self, x):
        # noinspection PyUnresolvedReferences
        return autograd.numpy.tan(autograd.numpy.pi * (x - 0.5)) * self.scale + self.loc

    def sample(self):
        return torch.distributions.Cauchy(self.loc, self.scale).rsample().item()

    def to_dict(self):
        return {"loc": self.loc, "scale": self.scale}

    @classmethod
    def from_dict(cls, dictionary):
        super(Cauchy, cls).from_dict(dictionary)
        return cls(dictionary["loc"], dictionary["scale"])

    @staticmethod
    def _validate(dictionary):
        assert (
            "loc" in dictionary and "scale" in dictionary and dictionary["scale"] >= 0.0
        )


class Uniform(Distribution):
    def __init__(self, low=0.0, high=1.0):

        self.low = low
        self.high = high

    def log_pdf(self, x):

        if self.low <= x <= self.high:
            # noinspection PyUnresolvedReferences
            return -autograd.numpy.log(self.high - self.low)

        # noinspection PyUnresolvedReferences
        return -autograd.numpy.inf

    def cdf(self, x):
        result = (x - self.low) / (self.high - self.low)
        # noinspection PyUnresolvedReferences
        return autograd.numpy.clip(result, 0.0, 1.0)

    def inverse_cdf(self, x):
        return x * (self.high - self.low) + self.low

    def sample(self):
        return torch.distributions.Uniform(self.low, self.high).rsample().item()

    def to_dict(self):
        return {"low": self.low, "high": self.high}

    @classmethod
    def from_dict(cls, dictionary):
        super(Uniform, cls).from_dict(dictionary)
        return cls(dictionary["low"], dictionary["high"])

    @staticmethod
    def _validate(dictionary):
        assert (
            "low" in dictionary
            and "high" in dictionary
            and dictionary["low"] < dictionary["high"]
        )


class HalfNormal(Normal):
    def __init__(self, scale):

        super().__init__(0.0, scale)

    def log_pdf(self, x):

        # noinspection PyUnresolvedReferences
        log_pdf = super(HalfNormal, self).log_pdf(x) + autograd.numpy.log(2)
        # noinspection PyUnresolvedReferences
        log_pdf = autograd.numpy.where(x >= 0.0, log_pdf, -autograd.numpy.inf)

        return log_pdf

    def cdf(self, x):
        return 2 * super(HalfNormal, self).cdf(x) - 1

    def inverse_cdf(self, x):
        return super(HalfNormal, self).inverse_cdf((x + 1) / 2)

    def sample(self):
        return torch.distributions.HalfNormal(self.scale).rsample().item()

    def to_dict(self):
        return {"scale": self.scale}

    @classmethod
    def from_dict(cls, dictionary):
        super(HalfNormal, cls).from_dict(dictionary)
        return cls(dictionary["scale"])

    @staticmethod
    def _validate(dictionary):
        assert "scale" in dictionary and dictionary["scale"] >= 0.0


class Gamma(Distribution):
    def __init__(self, alpha, rate):
        self.alpha = alpha
        self.rate = rate

    def log_pdf(self, x):
        # noinspection PyUnresolvedReferences
        return (
            self.alpha * autograd.numpy.log(self.rate)
            + (self.alpha - 1) * autograd.numpy.log(x)
            - self.rate * x
            - autograd.scipy.special.gamma(self.alpha)
        )

    def cdf(self, x):
        raise NotImplementedError()

    def inverse_cdf(self, x):
        raise NotImplementedError()

    def sample(self):
        return torch.distributions.Gamma(self.alpha, self.rate).rsample().item()

    def to_dict(self):
        return {"alpha": self.alpha, "rate": self.rate}

    @classmethod
    def from_dict(cls, dictionary):
        super(Gamma, cls).from_dict(dictionary)
        return cls(dictionary["alpha"], dictionary["rate"])

    @staticmethod
    def _validate(dictionary):
        assert (
            "alpha" in dictionary
            and dictionary["alpha"] >= 0.0
            and "rate" in dictionary
            and dictionary["rate"] >= 0.0
        )

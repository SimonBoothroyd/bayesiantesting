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


class Distribution(abc.ABC):
    @abc.abstractmethod
    def log_pdf(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def cdf(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse_cdf(self, x):
        raise NotImplementedError()


class Exponential(Distribution):
    def __init__(self, rate):
        self.rate = rate

    def log_pdf(self, x):

        if x < 0.0:
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


class Uniform(Distribution):
    def __init__(self, low=0.0, high=1.0):

        self.low = low
        self.high = high

    def log_pdf(self, x):

        if self.low <= x <= self.high:
            # noinspection PyUnresolvedReferences
            return -autograd.numpy.log(self.high - self.low)

        return -autograd.numpy.inf

    def cdf(self, x):
        result = (x - self.low) / (self.high - self.low)
        # noinspection PyUnresolvedReferences
        return autograd.numpy.clip(result, 0.0, 1.0)

    def inverse_cdf(self, x):
        return x * (self.high - self.low) + self.low

    def sample(self):
        return torch.distributions.Uniform(self.low, self.high).rsample().item()


class HalfNormal(Normal):

    def __init__(self, scale):

        super().__init__(0.0, scale)

    def log_pdf(self, x):

        log_pdf = super(HalfNormal, self).log_pdf(x) + autograd.numpy.log(2)
        log_pdf = autograd.numpy.where(x >= 0.0, log_pdf, -autograd.numpy.inf)

        return log_pdf

    def cdf(self, x):
        return 2 * super(HalfNormal, self).cdf(x) - 1

    def inverse_cdf(self, x):
        return super(HalfNormal, self).inverse_cdf((x + 1) / 2)

    def sample(self):
        return torch.distributions.HalfNormal(self.scale).rsample().item()


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

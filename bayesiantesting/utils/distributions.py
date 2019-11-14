"""
Differentiable distributions
"""
import autograd
import autograd.numpy
import autograd.scipy.special
import autograd.scipy.stats.gamma


def exponential_cdf(x, rate):
    return 1 - autograd.numpy.exp(-rate * x)


def exponential_inverse_cdf(x, rate):
    return -autograd.numpy.log(1 - x) / rate


def normal_cdf(x, loc, scale):
    return 0.5 * (
        1
        + autograd.scipy.special.erf((x - loc) * (1.0 / scale) / autograd.numpy.sqrt(2))
    )


def normal_inverse_cdf(x, loc, scale):
    return loc + scale * autograd.scipy.special.erfinv(2 * x - 1) * autograd.numpy.sqrt(
        2
    )


def half_normal_cdf(x, scale):
    return 2 * normal_cdf(x, 0.0, scale) - 1


def half_normal_inverse_cdf(x, scale):
    return normal_inverse_cdf((x + 1) / 2, 0.0, scale)

"""
This module implements a Hamiltonian and a No-U-Turn Sampler (NUTS)
sampler, which uses a Leapfrog integrator.

The code here is based on that in the the sampyl repository

https://github.com/mcleonard/sampyl/blob/master/sampyl/

but modified to avoid their state object, reformatted with black,
made flake8 compliant, and had docstrings converted to Numpy style.

## Original
:copyright: (c) 2015 by Mat Leonard.
:license: MIT, see LICENSE for more details.
"""
import autograd
import numpy
import torch

from bayesiantesting.kernels.samplers import Sampler


class Hamiltonian(Sampler):
    """Hamiltonian MCMC sampler."""

    def __init__(self, log_p_function, model_collection, step_size=1.0, n_steps=5):
        """
        Parameters
        ----------
        step_size: float
            The step size for the deterministic proposals.
        n_steps: int
            The number of leapfrog steps to take per step.
        """

        super().__init__(log_p_function, model_collection)

        if model_collection.n_models > 1:
            raise NotImplementedError()

        self._step_size = numpy.ones(model_collection.n_models) * step_size

        for index, model in enumerate(model_collection.models):
            self._step_size[index] /= model.n_trainable_parameters ** (1 / 4)

        self._n_steps = n_steps

    @staticmethod
    def _energy(log_p, momentum):
        return log_p - 0.5 * numpy.dot(momentum, momentum)

    @staticmethod
    def _leapfrog(parameters, model_index, momentum, gradient_function, step_size):

        r1 = momentum + step_size / 2 * gradient_function(parameters, model_index)
        x1 = parameters + step_size * r1
        r2 = r1 + step_size / 2 * gradient_function(x1, model_index)

        return x1, r2

    def _initial_momentum(self, model_index):
        model = self._model_collection[model_index]
        return torch.randn(model.n_trainable_parameters).item()

    def step(self, parameters, model_index, log_p, adapt=False):

        initial_momentum = self._initial_momentum(model_index)
        proposed_parameters, proposed_momentum = parameters, initial_momentum

        # Propose the new parameters.
        for i in range(self._n_steps):

            proposed_parameters, proposed_momentum = self._leapfrog(
                parameters=proposed_parameters,
                model_index=model_index,
                momentum=proposed_momentum,
                gradient_function=self._gradient_function,
                step_size=self._step_size[model_index],
            )

        proposed_log_p = self._log_p_function(proposed_parameters, model_index)

        # Decide whether to accept the move or not.
        energy_new = self._energy(proposed_log_p, proposed_momentum)
        energy = self._energy(log_p, initial_momentum)

        criteria = numpy.min(numpy.array([0, energy_new - energy]))

        random_number = numpy.log(torch.rand((1,)).item())
        accept = random_number < criteria

        # Update the bookkeeping
        self._proposed_moves[model_index] += 1

        if accept:

            self._accepted_moves[model_index] += 1

            parameters = proposed_parameters
            log_p = proposed_log_p

        return parameters, log_p, accept


class NUTS(Hamiltonian):
    """ No-U-Turn sampler (Hoffman & Gelman, 2014) for sampling from a
    probability distribution defined by a log P(theta) function.

    For technical details, see the paper:
    http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf
    """

    def __init__(
        self,
        log_p_function,
        model_collection,
        step_size=0.01,
        energy_max=1000.0,
        target_accept=0.65,
        gamma=0.05,
        k=0.75,
        t0=10.0,
    ):
        """
        Parameters
        ----------
        step_size: float
            Initial step size for the deterministic proposals. If None,
            a value will be chosen automatically.
        energy_max: float
            Maximum energy.
        target_accept: float
            Target acceptance rate.
        gamma: float
            The value of gamma.
        k: float
            Scales the speed of step size adaptation.
        t0: float
            Slows initial step size adaptation.
        """

        super().__init__(log_p_function, model_collection, step_size, -1)

        self._energy_max = energy_max

        self._target_accept = target_accept
        self._gamma = gamma
        self._k = k
        self._t0 = t0

        self._h_bar = 0.0
        self._e_bar = 1.0
        self._mu = numpy.log(self._step_size * 10)

        self._sampled = 0

        self._has_adapted = False

    @staticmethod
    def find_reasonable_epsilon(parameters, log_p_function):
        """Heuristic for choosing an initial value of epsilon
        (Algorithm 4)."""

        gradient_function = autograd.grad(log_p_function)

        x, x0 = parameters, parameters

        e0 = 1.0
        r0 = torch.randn(len(x)).item()

        x1, r1 = Hamiltonian._leapfrog(x0, 0, r0, gradient_function, e0)

        log_p_0 = log_p_function(x1)
        log_p_gradient_0 = gradient_function(x1, 0)

        log_p = log_p_0
        log_p_gradient = log_p_gradient_0

        # Get into a regime where things are finite / not NaN using the
        while numpy.isinf(log_p) or any(numpy.isinf(log_p_gradient)):

            e0 *= 0.5

            x1, r1 = Hamiltonian._leapfrog(x0, 0, r0, gradient_function, e0)

            log_p = log_p_function(x1)
            log_p_gradient = gradient_function(x1, 0)

        log_alpha = log_p - log_p_0 - 0.5 * (numpy.dot(r1, r1) - numpy.dot(r0, r0))

        accept = 1.0 if log_alpha > numpy.log(0.5) else -1.0

        while accept * log_alpha > -accept * numpy.log(2):

            e0 = e0 * (2.0 ** accept)

            x1, r1 = Hamiltonian._leapfrog(x0, 0, r0, gradient_function, e0)

            log_p = log_p_function(x1)
            log_alpha = log_p - log_p_0 - 0.5 * (numpy.dot(r1, r1) - numpy.dot(r0, r0))

        return e0

    @staticmethod
    def bern(p):
        return torch.rand((1,)).item() < p

    def step(self, parameters, model_index, log_p, adapt=False):

        x = parameters
        r0 = self._initial_momentum(model_index)

        u = torch.rand((1,)).item()
        e = self._step_size[model_index]

        xn, xp, rn, rp, y = x, x, r0, r0, x
        j, n, s = 0, 1, 1
        a, na = 0, 0

        while s == 1:

            # Choose a direction vj ∼ Uniform({−1, 1}
            v = self.bern(0.5) * 2 - 1

            if v == -1:
                xn, rn, _, _, x1, n1, s1, a, na = self._build_tree(
                    xn, model_index, rn, u, v, j, e, x, r0,
                )
            else:
                _, _, xp, rp, x1, n1, s1, a, na = self._build_tree(
                    xp, model_index, rp, u, v, j, e, x, r0,
                )

            if s1 == 1 and self.bern(numpy.min(numpy.array([1, n1 / n]))):
                y = x1

            dx = xp - xn
            s = s1 * (numpy.dot(dx, rn) >= 0) * (numpy.dot(dx, rp) >= 0)
            n = n + n1
            j = j + 1

        if not adapt:

            self._step_size[model_index] = self._e_bar

        else:

            # Adapt step size
            m = self._sampled + 1
            w = 1.0 / (m + self._t0)

            self._h_bar = (1 - w) * self._h_bar + w * (self._target_accept - a / na)
            log_e = self._mu - (m ** 0.5 / self._gamma) * self._h_bar
            self._step_size = numpy.exp(log_e)
            z = m ** (-self._k)
            self._e_bar = numpy.exp(z * log_e + (1 - z) * numpy.log(self._e_bar))

        # Update the bookkeeping
        self._proposed_moves[model_index] += 1
        self._accepted_moves[model_index] += 1

        proposed_log_p = self._log_p_function(y, model_index)

        return y, proposed_log_p, True

    def _build_tree(self, x, model_index, r, u, v, j, e, x0, r0):

        if j == 0:
            # Base case — take one leapfrog step in the direction v
            x1, r1 = self._leapfrog(x, model_index, r, self._gradient_function, v * e)
            energy = self._energy(x1, r1)
            energy_0 = self._energy(x0, r0)
            delta_energy = energy - energy_0

            n1 = numpy.log(u) - delta_energy <= 0
            s1 = numpy.log(u) - delta_energy < self._energy_max

            return (
                x1,
                r1,
                x1,
                r1,
                x1,
                n1,
                s1,
                numpy.min(numpy.array([1, numpy.exp(delta_energy)])),
                1,
            )
        else:
            # Recursion — implicitly build the left and right subtrees
            xn, rn, xp, rp, x1, n1, s1, a1, na1 = self._build_tree(
                x, model_index, r, u, v, j - 1, e, x0, r0
            )
            if s1 == 1:
                if v == -1:
                    xn, rn, _, _, x2, n2, s2, a2, na2 = self._build_tree(
                        xn, model_index, rn, u, v, j - 1, e, x0, r0,
                    )
                else:
                    _, _, xp, rp, x2, n2, s2, a2, na2 = self._build_tree(
                        xp, model_index, rp, u, v, j - 1, e, x0, r0,
                    )
                if NUTS.bern(n2 / max(n1 + n2, 1.0)):
                    x1 = x2

                a1 = a1 + a2
                na1 = na1 + na2

                dx = xp - xn
                s1 = s2 * (numpy.dot(dx, rn) >= 0) * (numpy.dot(dx, rp) >= 0)
                n1 = n1 + n2
            return xn, rn, xp, rp, x1, n1, s1, a1, na1

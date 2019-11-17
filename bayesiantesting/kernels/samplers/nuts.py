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
import autograd.numpy as np


class Hamiltonian:
    """Hamiltonian MCMC sampler."""

    def __init__(self, log_p, n_parameters, step_size=1, n_steps=5):
        """Initializes self.

        Parameters
        ----------
        log_p: function
            The log probability function to sample.
        n_parameters: int
            The number of parameters to sample over.
        step_size: float
            The step size for the deterministic proposals.
        n_steps: int
            The number of leapfrog steps to take per step.
        """

        self._step_size = step_size / n_parameters ** (1 / 4)
        self._n_steps = n_steps

        self._scale = np.ones(n_parameters)

        self._log_p = log_p
        self._gradient = autograd.grad(log_p)

    @staticmethod
    def leapfrog(x, r, step_size, grad):

        r1 = r + step_size / 2 * grad(x)
        x1 = x + step_size * r1
        r2 = r1 + step_size / 2 * grad(x1)

        return x1, r2

    @staticmethod
    def accept(x, y, r_0, r, log_p):

        energy_new = Hamiltonian.energy(log_p, y, r)
        energy = Hamiltonian.energy(log_p, x, r_0)
        criteria = np.min(np.array([0, energy_new - energy]))

        return np.log(np.random.rand()) < criteria and not np.isnan(criteria)

    @staticmethod
    def energy(log_p, x, r):
        return log_p(x) - 0.5 * np.dot(r, r)

    @staticmethod
    def initial_momentum(scale):
        return np.random.normal(0, scale)

    def step(self, parameters, adapt=False):

        x = parameters
        r0 = self.initial_momentum(self._scale)
        y, r = x, r0

        for i in range(self._n_steps):
            y, r = Hamiltonian.leapfrog(y, r, self._step_size, self._gradient)

        accepted = False

        if Hamiltonian.accept(x, y, r0, r, self._log_p):

            x = y
            accepted = True

        return x, accepted


class NUTS(Hamiltonian):
    """ No-U-Turn sampler (Hoffman & Gelman, 2014) for sampling from a
    probability distribution defined by a log P(theta) function.

    For technical details, see the paper:
    http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf
    """

    def __init__(
        self,
        log_p,
        n_parameters,
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
        gamma
        k: float
            Scales the speed of step size adaptation.
        t0: float
            Slows initial step size adaptation.
        """

        super(NUTS, self).__init__(log_p, n_parameters, step_size, -1)

        self._energy_max = energy_max

        self._target_accept = target_accept
        self._gamma = gamma
        self._k = k
        self._t0 = t0

        self._h_bar = 0.0
        self._e_bar = 1.0
        self._mu = np.log(self._step_size * 10)

        self._sampled = 0

        self._has_adapted = False

    @staticmethod
    def find_reasonable_epsilon(parameters, log_p_function):
        """Heuristic for choosing an initial value of epsilon
        (Algorithm 4)."""
        gradient_function = autograd.grad(log_p_function)

        x, x0 = parameters, parameters

        e0 = 1.0
        r0 = np.random.normal(0.0, 1.0, len(x))

        x1, r1 = Hamiltonian.leapfrog(x0, r0, e0, gradient_function)

        log_p_0 = log_p_function(x1)
        log_p_gradient_0 = gradient_function(x1)

        log_p = log_p_0
        log_p_gradient = log_p_gradient_0

        # Get into a regime where things are finite / not NaN using the
        while (
            np.isinf(log_p)
            or np.isnan(log_p)
            or any(np.isinf(log_p_gradient))
            or any(np.isnan(log_p_gradient))
        ):

            e0 *= 0.5

            x1, r1 = Hamiltonian.leapfrog(x0, r0, e0, gradient_function)

            log_p = log_p_function(x1)
            log_p_gradient = gradient_function(x1)

        log_alpha = log_p - log_p_0 - 0.5 * (np.dot(r1, r1) - np.dot(r0, r0))

        accept = 1.0 if log_alpha > np.log(0.5) else -1.0

        while accept * log_alpha > -accept * np.log(2):

            e0 = e0 * (2.0 ** accept)

            x1, r1 = Hamiltonian.leapfrog(x0, r0, e0, gradient_function)

            log_p = log_p_function(x1)

            log_alpha = log_p - log_p_0 - 0.5 * (np.dot(r1, r1) - np.dot(r0, r0))

        return e0

    def step(self, parameters, adapt=False):

        if adapt:
            self._has_adapted = True
        elif not self._has_adapted:
            raise ValueError("This sampler must be adaptively stepped at least once.")

        x = parameters
        r0 = self.initial_momentum(self._scale)

        u = np.random.uniform()
        e = self._step_size

        xn, xp, rn, rp, y = x, x, r0, r0, x
        j, n, s = 0, 1, 1
        a, na = 0, 0

        while s == 1:

            # Choose a direction vj ∼ Uniform({−1, 1}
            v = self.bern(0.5) * 2 - 1

            if v == -1:
                xn, rn, _, _, x1, n1, s1, a, na = self.build_tree(
                    xn,
                    rn,
                    u,
                    v,
                    j,
                    e,
                    x,
                    r0,
                    self._log_p,
                    self._gradient,
                    self._energy_max,
                )
            else:
                _, _, xp, rp, x1, n1, s1, a, na = self.build_tree(
                    xp,
                    rp,
                    u,
                    v,
                    j,
                    e,
                    x,
                    r0,
                    self._log_p,
                    self._gradient,
                    self._energy_max,
                )

            if s1 == 1 and self.bern(np.min(np.array([1, n1 / n]))):
                y = x1

            dx = xp - xn
            s = s1 * (np.dot(dx, rn) >= 0) * (np.dot(dx, rp) >= 0)
            n = n + n1
            j = j + 1

        if not adapt:

            self._step_size = self._e_bar

        else:

            # Adapt step size
            m = self._sampled + 1
            w = 1.0 / (m + self._t0)

            h_bar_new = (1 - w) * self._h_bar + w * (self._target_accept - a / na)
            log_e = self._mu - (m ** 0.5 / self._gamma) * h_bar_new
            step_size_new = np.exp(log_e)

            if not np.isnan(step_size_new):
                self._h_bar = h_bar_new
                self._step_size = step_size_new

                z = m ** (-self._k)
                self._e_bar = np.exp(z * log_e + (1 - z) * np.log(self._e_bar))

        print(y)
        return y, True

    @staticmethod
    def bern(p):
        return np.random.uniform() < p

    @staticmethod
    def build_tree(x, r, u, v, j, e, x0, r0, log_p, log_p_gradient, energy_max):

        if j == 0:
            # Base case — take one leapfrog step in the direction v
            x1, r1 = Hamiltonian.leapfrog(x, r, v * e, log_p_gradient)
            energy = Hamiltonian.energy(log_p, x1, r1)
            energy_0 = Hamiltonian.energy(log_p, x0, r0)
            delta_energy = energy - energy_0

            n1 = np.log(u) - delta_energy <= 0
            s1 = np.log(u) - delta_energy < energy_max and not np.isnan(energy) and not np.isinf(energy)
            return (
                x1,
                r1,
                x1,
                r1,
                x1,
                n1,
                s1,
                np.min(np.array([1, np.exp(delta_energy)])),
                1,
            )
        else:
            # Recursion — implicitly build the left and right subtrees
            xn, rn, xp, rp, x1, n1, s1, a1, na1 = NUTS.build_tree(
                x, r, u, v, j - 1, e, x0, r0, log_p, log_p_gradient, energy_max
            )
            if s1 == 1:
                if v == -1:
                    xn, rn, _, _, x2, n2, s2, a2, na2 = NUTS.build_tree(
                        xn,
                        rn,
                        u,
                        v,
                        j - 1,
                        e,
                        x0,
                        r0,
                        log_p,
                        log_p_gradient,
                        energy_max,
                    )
                else:
                    _, _, xp, rp, x2, n2, s2, a2, na2 = NUTS.build_tree(
                        xp,
                        rp,
                        u,
                        v,
                        j - 1,
                        e,
                        x0,
                        r0,
                        log_p,
                        log_p_gradient,
                        energy_max,
                    )
                if NUTS.bern(n2 / max(n1 + n2, 1.0)):
                    x1 = x2

                a1 = a1 + a2
                na1 = na1 + na2

                dx = xp - xn
                s1 = s2 * (np.dot(dx, rn) >= 0) * (np.dot(dx, rp) >= 0)
                n1 = n1 + n2
            return xn, rn, xp, rp, x1, n1, s1, a1, na1

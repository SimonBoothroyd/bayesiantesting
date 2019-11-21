import arviz
import autograd
import corner
import numpy
import numpy as np
import torch
import bayesiantesting.utils.distributions as distributions
import scipy.optimize
from matplotlib import pyplot


class Model:
    """ Sets up a simply model based on the user-specified prior
    types and parameters
    """

    @property
    def name(self):
        """str: The name of this model."""
        return self._name

    @property
    def priors(self):
        return self._priors

    @property
    def fixed_parameters(self):
        return self._fixed_parameters

    @property
    def n_trainable_parameters(self):
        """int: The number of trainable parameters within this model."""
        return len(self._prior_labels)

    @property
    def trainable_parameter_labels(self):
        """list of str: The friendly names of the parameters which are allowed to vary."""
        return self._prior_labels

    @property
    def n_fixed_parameters(self):
        """int: The number of fixed parameters within this model."""
        return len(self._fixed_labels)

    @property
    def fixed_parameter_labels(self):
        """list of str: The friendly names of the parameters which are fixed."""
        return self._fixed_labels

    @property
    def n_total_parameters(self):
        """int: The total number of parameters within this model."""
        return self.n_trainable_parameters + self.n_fixed_parameters

    @property
    def all_parameter_labels(self):
        """list of str: The friendly names of the parameters within this model."""
        return self._prior_labels + self._fixed_labels

    def __init__(self, name, priors, fixed_parameters):
        """Constructs a new `MCMCModel` object.

        Parameters
        ----------
        name: str
            The name of this model.
        priors: dict of str and tuple of float
            The settings for each of the priors, whose keys are the friendly
            name of the parameter associated with the prior. There should be
            one entry per trainable parameter.
        fixed_parameters: dict of str and float
            The values of the fixed model parameters, whose keys of the name
            associated with the parameter.
        """
        self._name = name

        self._priors = []
        self._prior_labels = []

        self._fixed_parameters = []
        self._fixed_labels = []

        for parameter_name in priors:

            self._priors.append(self._initialize_prior(priors[parameter_name]))
            self._prior_labels.append(parameter_name)

        for parameter_name in fixed_parameters:

            self._fixed_parameters.append(fixed_parameters[parameter_name])
            self._fixed_labels.append(parameter_name)

        common_parameters = set(self._fixed_labels).intersection(
            set(self._prior_labels)
        )

        if len(common_parameters) > 0:

            raise ValueError(
                f"The {', '.join(common_parameters)} have been flagged "
                f"as being both fixed and trainable."
            )

    @staticmethod
    def _initialize_prior(settings):

        prior_type, prior_values = settings

        if prior_type == "exponential":

            if not np.isclose(prior_values[0], 0.0):
                # The loc argument is not supported in PyTorch.
                raise NotImplementedError()

            prior = distributions.Exponential(rate=1.0 / prior_values[1])

        elif prior_type == "gamma":

            if not np.isclose(prior_values[1], 0.0):
                # The loc argument is not supported in PyTorch.
                raise NotImplementedError()

            prior = distributions.Gamma(prior_values[0], rate=1.0 / prior_values[2])

        elif prior_type == "normal":

            prior = distributions.Normal(prior_values[0], prior_values[1])

        elif prior_type == "uniform":

            prior = distributions.Uniform(prior_values[0], prior_values[1])

        else:
            raise NotImplementedError()

        return prior

    def sample_priors(self):
        """Generates a set of random parameters from the prior
        distributions. Those parameters without a prior will be
        assigned their fixed values.

        Returns
        -------
        numpy.ndarray:
            The sampled parameters with shape=(`n_trainable_parameters`).
        """

        initial_parameters = np.zeros(self.n_trainable_parameters)

        for index, prior in enumerate(self._priors):
            initial_parameters[index] = prior.sample()

        return initial_parameters

    def find_maximum_a_posteriori(
        self, initial_parameters=None, optimisation_method="L-BFGS-B"
    ):
        """ Find the maximum a posteriori of the posterior by doing a simple
        minimisation.

        Parameters
        ---------
        initial_parameters: numpy.ndarray, optional
            The initial values to start from when doing the minimisation
            with shape=(n_trainable_parameters). If None, these values are
            sampled randomly from the priors.
        optimisation_method: str
            The optimizing method to use.
        """

        if initial_parameters is None:
            initial_parameters = self.sample_priors()[0 : self.n_trainable_parameters]

        if len(initial_parameters) != self.n_trainable_parameters:

            raise ValueError(
                "The initial parameters must have a length "
                "equal to the number of parameters to train."
            )

        # Define the function to minimize.
        def negative_log_posterior(x):
            return -self.evaluate_log_posterior(x)

        gradient_function = autograd.grad(negative_log_posterior)

        results = scipy.optimize.minimize(
            fun=negative_log_posterior,
            x0=initial_parameters,
            jac=gradient_function,
            method=optimisation_method,
        )

        return numpy.array(results.x)

    def evaluate_log_prior(self, parameters):
        """Evaluates the log value of the prior for a
        set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=n_parameters)
            to evaluate at.

        Returns
        -------
        float
            The sum of the log values of priors evaluated at `parameters`.
        """
        log_prior = 0.0

        for index, prior in enumerate(self._priors):
            log_prior += prior.log_pdf(parameters[index])

        return log_prior

    def evaluate_log_likelihood(self, parameters):
        """Evaluates the log value of the this models likelihood for
        a set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=n_parameters)
            to evaluate at.

        Returns
        -------
        float
            The log value of the likelihood evaluated at `parameters`.
        """
        return 0.0

    def evaluate_log_posterior(self, parameters):
        """Evaluates the *unnormalized* log posterior for
        a set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=n_parameters)
            to evaluate at.

        Returns
        -------
        float
            The log value of the posterior evaluated at `parameters`.
        """
        return self.evaluate_log_prior(parameters) + self.evaluate_log_likelihood(
            parameters
        )

    def compute_percentage_deviations(self, parameters):
        """Computes the deviation of this models predicted
        values from the measured data it is being conditioned
        upon.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=n_parameters)
            to evaluate at.

        Returns
        -------
        dict of str and numpy.ndarray

        """
        raise NotImplementedError()

    def plot_trace(self, trace, show=False):
        """Use `Arviz` to plot a trace of the trainable parameters,
        alongside a histogram of their distribution.

        Parameters
        ----------
        trace: numpy.ndarray
            The parameter trace with shape=(n_steps, n_trainable_parameters+1)
        show: bool
            If true, the plot will be shown.

        Returns
        -------
        matplotlib.pyplot.Figure
            The plotted figure.
        """

        trace_dict = {}

        for index, label in enumerate(self._prior_labels):
            trace_dict[label] = trace[:, index + 1]

        data = arviz.convert_to_inference_data(trace_dict)

        axes = arviz.plot_trace(data)
        figure = axes[0][0].figure

        if show:
            figure.show()

        return figure

    def plot_corner(self, trace, show=False):
        """Use `corner` to plot a corner plot of the parameter
        distributions.

        Parameters
        ----------
        trace: numpy.ndarray
            The parameter trace with shape=(n_steps, n_trainable_parameters+1)
        show: bool
            If true, the plot will be shown.

        Returns
        -------
        matplotlib.pyplot.Figure
            The plotted figure.
        """

        figure = corner.corner(
            trace[:, 1 : 1 + len(self._prior_labels)],
            labels=self._prior_labels,
            color="#17becf",
        )

        if show:
            figure.show()

        return figure

    def plot_log_p(self, log_p, show=False, label="$log p$"):
        """Plot the log p trace.

        Parameters
        ----------
        log_p: numpy.ndarray
            The log p trace with shape=(n_steps, 1)
        show: bool
            If true, the plot will be shown.
        label: str
            The y-axis label to use.

        Returns
        -------
        matplotlib.pyplot.Figure
            The plotted figure.
        """
        figure, axes = pyplot.subplots(1, 1, figsize=(5, 5), dpi=200)

        axes.plot(log_p, color="#17becf")
        axes.set_title(f"{self._name}")
        axes.set_xlabel("steps")
        axes.set_ylabel(f"{label}")

        if show:
            figure.show()

        return figure

    def plot_percentage_deviations(self, percentage_deviations, show=False):
        """Plot the trace of the deviations of the trained model
        from the reference data.

        Parameters
        ----------
        percentage_deviations: dict of str and numpy.ndarray
            The deviations, whose values are arrays with shape=(n_steps, 1)
        show: bool
            If true, the plot will be shown.

        Returns
        -------
        matplotlib.pyplot.Figure
            The plotted figure.
        """

        figure, axes = pyplot.subplots(1, 1, figsize=(5, 5), dpi=200)

        for property_label in percentage_deviations:
            axes.plot(percentage_deviations[property_label], label=property_label.value)

        axes.set_xlabel("steps")
        axes.set_ylabel("%")

        axes.set_title(f"{self._name} Percentage Deviations")

        if len(percentage_deviations) > 0:

            axes.legend(
                loc="center",
                bbox_to_anchor=(0.5, -0.2),
                ncol=min(len(percentage_deviations), 3),
            )

        if show:
            figure.show()

        return figure

    def plot(self, trace, log_p, percentage_deviations, show=False):
        """Produce plots of this models traces. This is equivalent to
        calling `plot_trace`, `plot_corner`, `plot_log_p`,
        `plot_percentage_deviations`.

        Parameters
        ----------
        trace: numpy.ndarray
            The parameter trace with shape=(n_steps, n_trainable_parameters+1)
        log_p: numpy.ndarray
            The log p trace with shape=(n_steps, 1)
        percentage_deviations: dict of str and numpy.ndarray
            The deviations, whose values are arrays with shape=(n_steps, 1)
        show: bool
            If true, the plots will be shown.

        Returns
        -------
        tuple of matplotlib.pyplot.Figure
            The plotted figures.
        """
        return (
            self.plot_trace(trace, show),
            self.plot_corner(trace, show),
            self.plot_log_p(log_p, show),
            self.plot_percentage_deviations(percentage_deviations, show),
        )


class ModelCollection:
    """Represents a collection of models to simultaneously optimize.
    """

    @property
    def name(self):
        """str: The name of this model."""
        return self._name

    @property
    def models(self):
        """tuple of Model: The models which belong to this collection."""
        return self._models

    @property
    def n_models(self):
        """int: The number models which belong to this collection."""
        return len(self._models)

    def __init__(self, name, models):
        """Initializes self.

        Parameters
        ----------
        name: str
            The name of this collection.
        models: List of Model
            The models which belong to this collection.
        """

        # Make sure there are no models with duplicate names.
        assert len(set(model.name for model in models)) == len(models)

        self._name = name
        self._models = tuple(models)

        for model in self._models:

            if model.n_trainable_parameters <= 1 or all(
                isinstance(prior, (distributions.Exponential, distributions.Normal))
                for prior in model.priors
            ):
                continue

            raise ValueError("Currently only exponential priors are supported.")

    def _mapping_function(
        self, parameter, model_index_a, model_index_b, parameter_index
    ):
        """Attempts to map a given parameter from model a into a
        parameter in model b which yields a non-zero posterior
        probability.

        Parameters
        ----------
        parameter: float
            The value of the model a parameter.
        model_index_a: int
            The index of model a in this model collection.
        model_index_b: int
            The index of model b in this model collection.

        Returns
        -------
        float
            The mapped parameter.
        """

        model_a = self._models[model_index_a]
        model_b = self._models[model_index_b]

        if (
            parameter_index >= model_a.n_trainable_parameters
            and parameter_index >= model_b.n_trainable_parameters
        ):

            # These parameters aren't being trained so we don't need to
            # do any mapping so long as both models take the same fixed
            # value.
            if not numpy.isclose(
                model_a.fixed_parameters[
                    parameter_index - model_a.n_trainable_parameters
                ],
                model_b.fixed_parameters[
                    parameter_index - model_b.n_trainable_parameters
                ],
            ):

                raise NotImplementedError()

            return parameter

        elif (
            parameter_index < model_a.n_trainable_parameters
            and parameter_index < model_b.n_trainable_parameters
        ):

            cdf_x = model_a.priors[parameter_index].cdf(parameter)
            return model_b.priors[parameter_index].inverse_cdf(cdf_x)

        elif (
            model_a.n_trainable_parameters
            > parameter_index
            >= model_b.n_trainable_parameters
        ):

            # Handle the case where we are mapping to a model with a lower dimension.
            return model_a.priors[parameter_index].cdf(parameter)

        elif (
            model_a.n_trainable_parameters
            <= parameter_index
            < model_b.n_trainable_parameters
        ):

            # Handle the case where we are mapping to a model with a higher dimension.
            return model_b.priors[parameter_index].inverse_cdf(parameter)

        raise NotImplementedError()

    def map_parameters(self, parameters, model_index_a, model_index_b):
        """Attempts to map a set of trainable parameters from model
        a into a set of parameters with a non-zero posterior in model
        b.

        Parameters
        ----------
        parameters: numpy.ndarray
            The current parameters of model a, with shape=(model_a.n_trainable_parameters).
        model_index_a: int
            The index of model a in this model collection.
        model_index_b: int
            The index of model b in this model collection.

        Returns
        -------
        numpy.ndarray
            The current parameters of model a with any 'ghost' parameters
            added (shape=(model_b.n_total_parameters)).
        numpy.ndarray
            The mapped parameters with shape=(model_b.n_trainable_parameters).
        numpy.ndarray
            The jacobian associated with the mapping with
            shape=(model_b.n_trainable_parameters).
        """

        model_a = self._models[model_index_a]
        model_b = self._models[model_index_b]

        n_parameters = max(model_a.n_total_parameters, model_b.n_total_parameters)

        current_parameters = numpy.array([*parameters, *model_a.fixed_parameters])
        new_parameters = numpy.empty(n_parameters)

        jacobian_function = autograd.grad(self._mapping_function)
        jacobians = numpy.empty(n_parameters)

        if model_a.n_trainable_parameters < model_b.n_trainable_parameters:

            # If we are moving to a higher dimensional model, we
            # set the 'ghost' parameters to a random number drawn
            # from a uniform distribution.
            for j in range(
                model_a.n_trainable_parameters, model_b.n_trainable_parameters
            ):
                current_parameters[j] = torch.rand((1,)).item()

        for i in range(n_parameters):

            new_parameters[i] = self._mapping_function(
                current_parameters[i], model_index_a, model_index_b, i
            )
            jacobians[i] = jacobian_function(
                current_parameters[i], model_index_a, model_index_b, i
            )

        return (
            current_parameters,
            new_parameters[: model_b.n_trainable_parameters],
            jacobians,
        )

    def transition_probabilities(self, model_index_a, model_index_b):
        return 1.0

    def __len__(self):
        return self.n_models

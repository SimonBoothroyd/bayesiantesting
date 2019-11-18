import numpy

from bayesiantesting.kernels.rjmc import RJMCSimulation
from bayesiantesting.models import Model, ModelCollection
from bayesiantesting.utils import distributions
from matplotlib import pyplot


class GaussianModel(Model):
    """A representation of the two-center Lennard-Jones model, which
    can be evaluated using a surrogate model against a `NISTDataSet`.
    """

    def __init__(self, name, prior_settings, loc, scale):

        super().__init__(name, prior_settings, {})

        self._loc = loc
        self._scale = scale

    def evaluate_log_likelihood(self, parameters):
        return distributions.Normal(self._loc, self._scale).log_pdf(parameters)

    def compute_percentage_deviations(self, parameters):
        return {}


def main():

    priors_a = {"uniform": ("uniform", numpy.array([-5.0, 5.0]))}
    priors_b = {"uniform": ("uniform", numpy.array([5.0, 15.0]))}

    # Build the model / models.
    model_a = GaussianModel("gaussian_a", priors_a, 0.0, 1.0)
    model_b = GaussianModel("gaussian_b", priors_b, 10.0, 1.0)

    model_collection = ModelCollection('gaussians', [model_a, model_b])

    # Draw the initial parameter values from the model priors.
    initial_parameters = model_a.sample_priors()
    initial_model_index = 0

    simulation = RJMCSimulation(
        model_collection=model_collection,
        warm_up_steps=50000,
        steps=500000,
        discard_warm_up_data=True,
        swap_frequency=0.1,
    )

    trace, log_p_trace, percent_deviation_trace = simulation.run(
        initial_parameters, initial_model_index
    )

    for model_index in range(model_collection.n_models):

        model_trace_indices = trace[:, 0] == model_index

        model_trace = trace[model_trace_indices]
        model_log_p_trace = log_p_trace[model_trace_indices]

        model_collection.models[model_index].plot_trace(model_trace, show=True)
        model_collection.models[model_index].plot_corner(model_trace, show=True)
        model_collection.models[model_index].plot_log_p(model_log_p_trace, show=True)

    figure, axes = pyplot.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(trace[:, 0])
    axes[1].hist(trace[:, 0])

    axes[0].set_xlabel("Model Index")
    axes[1].set_xlabel("Model Index")

    figure.show()

    # Plot the output.
    pyplot.plot(log_p_trace)
    pyplot.show()


if __name__ == "__main__":
    main()

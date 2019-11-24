import numpy
import torch

from bayesiantesting.kernels.rjmc import BiasedRJMCSimulation
from bayesiantesting.models import Model, ModelCollection
from bayesiantesting.utils import distributions


class GaussianModel(Model):
    """A representation of the two-center Lennard-Jones model, which
    can be evaluated using a surrogate model against a `NISTDataSet`.
    """

    def __init__(self, name, prior_settings, loc, scale, weight=1.0):

        super().__init__(name, prior_settings, {})

        self._loc = loc
        self._scale = scale
        self._weight = weight

    def evaluate_log_likelihood(self, parameters):
        return numpy.log(self._weight) + distributions.Normal(
            self._loc, self._scale
        ).log_pdf(parameters)

    def compute_percentage_deviations(self, parameters):
        return {}


def main():

    random_seed = torch.randint(1000000, (1,)).item()

    torch.manual_seed(random_seed)
    numpy.random.seed(random_seed)

    print("==============================")
    print(f"Using a random seed of {random_seed}")
    print("==============================")

    priors_a = {"uniform": ("uniform", numpy.array([-5.0, 5.0]))}
    priors_b = {"uniform": ("uniform", numpy.array([5.0, 15.0]))}

    # Build the model / models.
    model_a = GaussianModel("gaussian_a", priors_a, 0.0, 1.0)
    model_b = GaussianModel("gaussian_b", priors_b, 10.0, 1.0, weight=0.5)

    model_collection = ModelCollection("gaussians", [model_a, model_b])

    # Draw the initial parameter values from the model priors.
    initial_parameters = model_a.sample_priors()
    initial_model_index = 0

    log_biases = numpy.log(numpy.array([1.0, 2.0]))

    simulation = BiasedRJMCSimulation(
        model_collection=model_collection,
        warm_up_steps=100000,
        steps=200000,
        discard_warm_up_data=True,
        output_directory_path="gaussian",
        swap_frequency=0.1,
        log_biases=log_biases,
    )

    simulation.run(initial_parameters, initial_model_index)


if __name__ == "__main__":
    main()
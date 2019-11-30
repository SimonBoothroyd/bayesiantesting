import corner
import numpy
from matplotlib import pyplot

from bayesiantesting.models.continuous import MultivariateGaussian
from bayesiantesting.utils.distributions import HalfNormal, Normal


def main():

    for model_name in ["UA", "AUA", "AUA+Q"]:

        trace = numpy.load(f"trace_{model_name}.npy")[::1000]

        n_parameters = trace.shape[1] - 1

        figure, axes = pyplot.subplots(1, n_parameters, figsize=(5, 5), dpi=200)

        print(f"{model_name}\n\n")

        for parameter_index in range(n_parameters):

            if parameter_index < 3:

                loc = numpy.mean(trace[:, parameter_index + 1])
                scale = numpy.std(trace[:, parameter_index + 1]) * 1.10

                distribution = Normal(loc, scale)
                x = numpy.linspace(-5 * scale + loc, 5 * scale + loc, 100)

                print(f"distributions.Normal({loc}, {scale})")

            else:

                scale = (
                    numpy.sqrt(
                        numpy.sum(trace[:, parameter_index + 1] ** 2)
                        / len(trace[:, parameter_index + 1])
                    )
                    * 1.10
                )

                distribution = HalfNormal(scale)
                x = numpy.linspace(0.0, 5 * scale, 100)

                print(f"distributions.HalfNormal({scale})")

            y = numpy.exp(distribution.log_pdf(x))

            axes[parameter_index].hist(trace[:, parameter_index + 1], density=True)
            axes[parameter_index].plot(x, y)

        figure.show()

        multivariate_mean = numpy.mean(trace[:, 1:], axis=0)
        multivariate_covariance = numpy.cov(trace[:, 1:].T)

        print(multivariate_mean, multivariate_covariance)

        mean_dictionary = {
            index: value for index, value in enumerate(multivariate_mean)
        }

        model = MultivariateGaussian(
            "gaussian", mean_dictionary, multivariate_covariance
        )

        trace = numpy.zeros((100000, len(multivariate_mean)))

        for i in range(len(trace)):
            trace[i] = model.sample_priors()

        figure = corner.corner(trace, color="#17becf",)

        figure.show()


if __name__ == "__main__":
    main()

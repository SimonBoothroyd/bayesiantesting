import pymc3
from matplotlib import pyplot


def main():

    with pymc3.Model() as gaussian_model:

        a = pymc3.Normal("a", mu=0, sigma=0.05)
        b = pymc3.Normal("b", mu=0, sigma=5000)

    with gaussian_model:

        step = pymc3.NUTS()
        trace = pymc3.sample(2000, step=step)

    pymc3.traceplot(trace)
    pyplot.show()


if __name__ == "__main__":
    main()

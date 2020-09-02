import numpy
import argparse
import functools
import json
import os
from multiprocessing.pool import Pool
import datetime

from studies.utilities import get_2clj_model, parse_input_yaml, prepare_data, fit_to_trace
from matplotlib import pyplot as plt
from scipy.stats import distributions as dist
from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.kernels.bayes import MBARIntegration
from bayesiantesting.models.continuous import UnconditionedModel
from bayesiantesting.models.discrete import TwoCenterLJModelCollection
from bayesiantesting.utils import distributions
from bayesiantesting.kernels.rjmc import BiasedRJMCSimulation
from studies.benchmarking import calculate_elpd, sample_trace, choose_test_datapoints, calculate_deviations, \
    plot_deviations


def main(compound, properties):


    # create file structure for output
    output_path = os.path.join('output',
    compound,
    properties,
    str(datetime.date.today()))
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'intermediate'), exist_ok=True)
    runfile_path = os.path.join('runfiles', compound, properties)
    os.makedirs(os.path.join(output_path, 'figures'), exist_ok=True)


    simulation_params = mcmc_choose_priors(runfile_path,output_path)
    print(simulation_params['priors'])
    bayes_factors, results = calculate_bayes_factor(simulation_params, runfile_path, output_path, n_processes=3)
    counts = rjmc_validation(simulation_params, results, runfile_path, output_path, n_processes=3)
    mcmc_benchmarking(simulation_params, runfile_path, output_path)
    print(results)
    print(bayes_factors)
    print(counts)


def mcmc_choose_priors(runfile_path, output_path):
    full_path = os.path.join(runfile_path, 'mcmc_basic_run.yaml')
    simulation_params = parse_input_yaml(full_path)
    # TODO clean up and merge runfiles
    # Load the data.
    data_set, property_types = prepare_data(simulation_params)

    # Set some initial parameter close to the MAP taken from
    # other runs of C2H6.
    # TODO add initial parameters to runfiles
    initial_parameters = {
        "UA": numpy.array([80, 0.375]),
        "AUA": numpy.array([80.0, 0.375, 0.13]),
        "AUA+Q": numpy.array([80.0, 0.375, 0.13, 0.4]),
    }
    model = get_2clj_model("AUA+Q", data_set, property_types, simulation_params)
    # Run the simulation.
    simulation = MCMCSimulation(
        model_collection=model, initial_parameters=initial_parameters[model.name]
    )

    path = os.path.join(output_path, 'intermediate', 'mcmc_prior')
    os.makedirs(path,exist_ok=True)
    trace, log_p_trace, percent_deviation_trace = simulation.run(
        warm_up_steps=int(simulation_params["steps"] * 0.2),
        steps=simulation_params["steps"],
        output_directory=path
    )

    params = simulation.fit_prior_exponential()
    params['epsilon'][1][1] *= 5
    params['sigma'][1][1] *= 5
    params['L'][1][1] *= 5
    params['Q'][1][1] *= 5
    variables = ['epsilon', 'sigma', 'L', 'Q']

    prior_figure_path = os.path.join(output_path,'figures','priors')
    os.makedirs(prior_figure_path,exist_ok=True)
    for i in range(len(trace[0, 1:])):
        plt.hist(trace[:, i+1], bins=50, alpha=0.5, color='b', label='Trace', density=True)
        x_vec = numpy.linspace(0.666*min(trace[:, i+1]), 1.5*max(trace[:, i+1]), num=500)
        if i == 3:
            plt.plot(x_vec, dist.expon.pdf(x_vec, *params[variables[i]][1]), label='Prior')
        else:
            plt.plot(x_vec, dist.norm.pdf(x_vec, *params[variables[i]][1]), label='Prior')
        plt.savefig(os.path.join(prior_figure_path, 'prior_'+variables[i]+'.png'))
    simulation_params["priors"] = params
    #   for key in simulation_params['priors'].keys()


    return simulation_params

def calculate_bayes_factor(simulation_params, runfile_path, output_path, n_processes):
    mbar_params = parse_input_yaml(os.path.join(runfile_path,"mbar_basic_run.yaml"))
    mbar_params['priors'] = simulation_params['priors']
    compound = simulation_params['compound']
    # Load the data.
    data_set, property_types = prepare_data(mbar_params, compound)

    # Build the models.
    models = [
        get_2clj_model(model_name, data_set, property_types, mbar_params)
        for model_name in ["UA", "AUA", "AUA+Q"]
    ]

    # Fit multivariate normal distributions to this models posterior.
    # This will be used as the analytical reference distribution.
    fitting_directory = os.path.join(output_path, "intermediate", "mbar_fitting")
    os.makedirs(fitting_directory, exist_ok=True)

    initial_parameters = {
        "UA": numpy.array([80, 0.375]),
        "AUA": numpy.array([80.0, 0.375, 0.13]),
        "AUA+Q": numpy.array([80.0, 0.375, 0.13, 0.4]),
    }

    with Pool(n_processes) as pool:
        all_fits = pool.map(
            functools.partial(
                fit_to_trace,
                warm_up_steps=mbar_params["fit_steps"],
                output_directory=fitting_directory,
                initial_parameters=initial_parameters,
            ),
            models,
        )

    # Run the MBAR calculations
    results = {}

    for model, fits in zip(models, all_fits):

        _, reference_model = fits

        # Run the MBAR simulation
        lambda_values = numpy.linspace(0.0, 1.0, 3)

        output_directory = os.path.join(output_path, f"mbar_{model.name}")

        simulation = MBARIntegration(
            lambda_values=lambda_values,
            model=model,
            steps=mbar_params["steps"],
            output_directory_path=output_directory,
            reference_model=reference_model,
        )

        _, integral, error = simulation.run(
            reference_model.sample_priors(), number_of_processes=n_processes
        )

        results[model.name] = {"integral": integral, "error": error}
    models = []
    evidences = []
    for key in results.keys():
        evidences.append(results[key]['integral'])
        models.append(key)
    evidences = numpy.asarray(evidences)
    evidences -= max(evidences)
    bayes_factors = numpy.exp(evidences)/min(numpy.exp(evidences))
    bayes_results = [models, bayes_factors]
    # TODO add errors in here


    with open(f"mbar_{compound}_results.json", "w") as file:
        json.dump(results, file, sort_keys=True, indent=4, separators=(",", ": "))

    return bayes_results, results

def rjmc_validation(simulation_params, results, runfile_path, output_path, n_processes):
    # Load in the simulation parameters.
    rjmc_params = parse_input_yaml(os.path.join(runfile_path, 'rjmc_basic_run.yaml'))
    rjmc_params['priors'] = simulation_params['priors']
    # Load the data.
    data_set, property_types = prepare_data(rjmc_params)

    # Build the model / models.
    sub_models = [
        get_2clj_model("AUA", data_set, property_types, rjmc_params),
        get_2clj_model("AUA+Q", data_set, property_types, rjmc_params),
        get_2clj_model("UA", data_set, property_types, rjmc_params),
    ]

    initial_parameters = {
        "UA": numpy.array([80, 0.375]),
        "AUA": numpy.array([80.0, 0.375, 0.13]),
        "AUA+Q": numpy.array([80.0, 0.375, 0.13, 0.4]),
    }

    # Load the mapping distributions
    mapping_distributions = []
    maximum_a_posteriori = []
    output_directory = os.path.join(output_path, 'intermediate', 'rjmc_fitting')
    os.makedirs(output_directory, exist_ok=True)
    with Pool(n_processes) as pool:

        pool.map(
            functools.partial(
                fit_to_trace,
                warm_up_steps=int(rjmc_params["fit_steps"] * 2.0 / 3.0),
                output_directory=output_directory,
                initial_parameters=initial_parameters,
            ),
            sub_models,
        )


    for model in sub_models:

        fit_path = os.path.join(output_path, 'intermediate', 'rjmc_fitting', f"{model.name}_univariate_fit.json")

        with open(fit_path) as file:
            fit_distributions = json.load(file)

        fit_model = UnconditionedModel(model.name, fit_distributions, {})
        mapping_distributions.append(fit_model.priors)

        # Determine the maximum a posteriori of the fit
        map_parameters = []

        for distribution in fit_model.priors:

            if isinstance(distribution, distributions.Normal):
                map_parameters.append(distribution.loc)
            else:
                raise NotImplementedError()

        maximum_a_posteriori.append(numpy.asarray(map_parameters))

    for mean, model in zip(maximum_a_posteriori, sub_models):
        print(model.evaluate_log_posterior(mean))

    # Create the full model collection
    model_collection = TwoCenterLJModelCollection(
        "2CLJ Models", sub_models, mapping_distributions
    )

    # Load in the bias factors
    # bias_file_name = f"mbar_{simulation_params['compound']}_results.json"
    # with open(bias_file_name) as file:
    #     bias_factor_dictionary = json.load(file)

    bias_factors = [
        results[model.name]["integral"] for model in sub_models
    ]
    bias_factors = -numpy.asarray(bias_factors)

    # Draw the initial parameter values from the model priors.
    initial_model_index = 1  # torch.randint(len(sub_models), (1,)).item()

    # initial_parameters = generate_initial_parameters(sub_models[initial_model_index])
    initial_parameters = maximum_a_posteriori[initial_model_index]

    output_directory_path = output_path

    simulation = BiasedRJMCSimulation(
        model_collection=model_collection,
        initial_parameters=initial_parameters,
        initial_model_index=initial_model_index,
        swap_frequency=rjmc_params["swap_freq"],
        log_biases=bias_factors,
    )

    trace, log_p_trace, percent_deviation_trace = simulation.run(
        warm_up_steps=int(rjmc_params["steps"] * 0.1),
        steps=rjmc_params["steps"],
        output_directory=output_directory_path,
    )
    rjmc_counts = numpy.zeros(3)
    for i in range(len(trace)):
        if trace[i, 0] == 0:
            rjmc_counts[0] += 1
        elif trace[i, 0] == 1:
            rjmc_counts[1] += 1
        elif trace[i, 0] == 2:
            rjmc_counts[2] += 1
    return rjmc_counts
def mcmc_placeholder(simulation_params):
    # Load the data.
    data_set, property_types = prepare_data(simulation_params)

    # Set some initial parameter close to the MAP taken from
    # other runs of C2H6.
    initial_parameters = {
        "UA": numpy.array([80, 0.375]),
        "AUA": numpy.array([80.0, 0.375, 0.13]),
        "AUA+Q": numpy.array([80.0, 0.375, 0.13, 0.4]),
    }
    model = get_2clj_model("AUA+Q", data_set, property_types, simulation_params)
    print(simulation_params['priors'])
    # Run the simulation.
    simulation = MCMCSimulation(
        model_collection=model, initial_parameters=initial_parameters[model.name]
    )


    trace, log_p_trace, percent_deviation_trace = simulation.run(
        warm_up_steps=int(simulation_params["steps"] * 0.2),
        steps=simulation_params["steps"],
    )
    model.plot(trace, log_p_trace, percent_deviation_trace, show=True)

def mcmc_benchmarking(simulation_params,runfile_path,output_path):

    # Load the data.
    data_set, property_types = prepare_data(simulation_params)

    # Set some initial parameter close to the MAP taken from
    # other runs of C2H6.
    initial_parameters = {
        "UA": numpy.array([80, 0.375]),
        "AUA": numpy.array([80.0, 0.375, 0.13]),
        "AUA+Q": numpy.array([80.0, 0.375, 0.13, 0.4]),
    }

    # Build the model / models.
    model_elpds = {}
    model_deviations = {}
    model_keys = ["UA", "AUA", "AUA+Q"]
    for model_name in model_keys:

        model = get_2clj_model(model_name, data_set, property_types, simulation_params)

        # Run the simulation.
        simulation = MCMCSimulation(
            model_collection=model, initial_parameters=initial_parameters[model.name]
        )
        path = os.path.join(output_path, 'intermediate', 'mcmc_benchmarking')
        os.makedirs(path, exist_ok=True)

        trace, log_p_trace, percent_deviation_trace = simulation.run(
            warm_up_steps=int(simulation_params["steps"] * 0.2),
            steps=simulation_params["steps"],
            output_directory=path
        )

        params = simulation.fit_prior_exponential()
        print(params)



        # calculate summary statistics
        benchmark_filepath = os.path.join(runfile_path, 'evaluate_params.yaml')
        test_set = choose_test_datapoints(benchmark_filepath, simulation_params, data_set)
        test_params = parse_input_yaml(benchmark_filepath)
        samples = sample_trace(trace, test_set, model_name, test_params["number_samples"])
        test_model = get_2clj_model(model_name, test_set, property_types, simulation_params)
        elpd, elppd = calculate_elpd(test_model, property_types, samples)

        #model.plot(trace, log_p_trace, percent_deviation_trace, show=True)

        model_elpds[model_name] = [elpd, elppd]
        fixed_samples = []
        for i in range(len(samples)):
            if model_name == "UA":
                fixed_samples.append(numpy.append(samples[i], [test_set.bond_length.magnitude / 10, 0]))
            elif model_name == "AUA":
                fixed_samples.append(numpy.append(samples[i], [0]))
            else:
                fixed_samples.append(samples[i])
        samples = numpy.asarray(fixed_samples)
        model_deviations[model_name] = calculate_deviations(property_types, test_set, data_set, samples)
    plot_deviations(model_deviations, model_keys, property_types, output_path)
    for property in property_types:
        print(property)
        for key in model_elpds.keys():
            print(f"For model {key},"
                  f" -ELPPD is {model_elpds[key][1][property]}"
                  f" tested against {len(model_elpds[key][0][property])} data points (average stdev away from experiment "
                  f"= {numpy.sqrt(2 * (model_elpds[key][1][property]/len(model_elpds[key][0][property])))}")
        print('==================')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Perform a calculation of Bayes factors and associated benchmarking for a compound.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--compound",
        "-c",
        type=str,
        help="The compound to compute the Bayes factors for.",
        required=True,
    )

    parser.add_argument(
        "--properties",
        "-p",
        type=str,
        help="The number processes to run the calculation across.",
        required=False,
        default=1,
    )

    args = parser.parse_args()
    main(args.compound, args.properties)

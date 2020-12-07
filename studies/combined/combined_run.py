import numpy
import argparse
import functools
import json
import os
import sys
from multiprocessing.pool import Pool
import datetime
import copy
# import shutil


from studies.utilities import get_2clj_model, parse_input_yaml, prepare_data, fit_to_trace, prior_dictionary_to_json
from matplotlib import pyplot as plt
import scipy.stats.distributions as dist
from bayesiantesting.kernels import MCMCSimulation
from bayesiantesting.kernels.bayes import MBARIntegration
from bayesiantesting.models.continuous import UnconditionedModel
from bayesiantesting.models.discrete import TwoCenterLJModelCollection
from bayesiantesting.utils import distributions
from bayesiantesting.kernels.rjmc import BiasedRJMCSimulation
from bayesiantesting.datasets.nist import NISTDataType
from studies.benchmarking import calculate_elpd, sample_trace, choose_test_datapoints, calculate_deviations, \
    plot_deviations


def main(compound, properties, output_location):
    # create file structure for output
    output_path = os.path.join(output_location,
                               'output',
                               compound,
                               properties,
                               str(datetime.date.today()))
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'intermediate'), exist_ok=True)
    runfile_path = os.path.join('runfiles', compound, properties)
    os.makedirs(os.path.join(output_path, 'figures'), exist_ok=True)

    simulation_params = mcmc_choose_priors(runfile_path, output_path)
    bayes_factors, results = calculate_bayes_factor(simulation_params, runfile_path, output_path, n_processes=3)
    counts = rjmc_validation(simulation_params, results, runfile_path, output_path, n_processes=3)
    mcmc_benchmarking(simulation_params, runfile_path, output_path)
    print(results)
    print(bayes_factors)


def mcmc_choose_priors(runfile_path, output_path):
    full_path = os.path.join(runfile_path, 'basic_run.yaml')
    simulation_params = parse_input_yaml(full_path)
    # TODO clean up and merge runfiles
    # Load the data.
    prior_data_set, property_types = prepare_data(simulation_params, scenario='prior')

    # TODO add initial parameters to runfiles
    initial_parameters = simulation_params['initial_parameters']
    model_keys = ['UA', 'AUA', 'AUA+Q']

    for property in property_types:
        print(property)
        print(prior_data_set.get_data(property))
    for model_name in model_keys:

        model = get_2clj_model(model_name, prior_data_set, property_types, simulation_params)

        simulation = MCMCSimulation(
            model_collection=model, initial_parameters=numpy.asarray(initial_parameters[model.name])
        )

        path = os.path.join(output_path, 'intermediate', 'mcmc_prior', model_name)
        os.makedirs(path, exist_ok=True)
        trace, log_p_trace, percent_deviation_trace = simulation.run(
            warm_up_steps=int(simulation_params["mcmc_steps"] * 0.2),
            steps=simulation_params["mcmc_steps"],
            output_directory=path
        )
        params_length = len(trace[0, 1:])
        params = simulation.fit_prior_exponential()
        params['epsilon'][1][1] *= 1
        params['sigma'][1][1] *= 1
        if model_name == 'UA':
            params_length -= 2
            variables = ['epsilon', 'sigma']
        elif model_name == 'AUA':
            params_length -= 1
            variables = ['epsilon', 'sigma', 'L']
            params['L'][1][1] *= 1
        elif model_name == 'AUA+Q':
            variables = ['epsilon', 'sigma', 'L', 'Q']
            if params['Q'][0] == 'exponential':
                params['Q'][1][1] *= 1
            # params['Q'][0] = 'uniform'
            # params['Q'][1][0] = numpy.float64(0.0)
            # params['Q'][1][1] = numpy.float64(5.0)
            # print(type(params['Q'][1][0]))
            # params['Q'] = ['uniform', numpy.asarray([0, 0.5])]
            # print(params['Q'])

        prior_figure_path = os.path.join(output_path, 'figures', 'priors', model_name)
        os.makedirs(prior_figure_path, exist_ok=True)

        for i in range(len(trace[0, 1:])):
            plt.clf()
            plt.hist(trace[:, i + 1], bins=50, alpha=0.5, color='b', label='Trace', density=True)
            x_vec = numpy.linspace(0.666 * min(trace[:, i + 1]), 1.5 * max(trace[:, i + 1]), num=500)
            if i == 3:
                if params['Q'][0] == 'gamma':
                    plt.plot(x_vec, dist.gamma.pdf(x_vec, params[variables[i]][1][0], scale=params[variables[i]][1][1]),
                             label='Prior')
                elif params['Q'][0] == 'exponential':
                    plt.plot(x_vec, dist.expon.pdf(x_vec, *params[variables[i]][1]), label='Prior')
            else:
                plt.plot(x_vec, dist.norm.pdf(x_vec, *params[variables[i]][1]), label='Prior')
            plt.legend()
            plt.savefig(os.path.join(prior_figure_path, 'prior_' + variables[i] + '.png'))

        simulation_params["priors"][model_name] = params
        prior_dictionary_to_json(
            simulation_params["priors"][model_name],
            os.path.join(path, "priors_" + model_name + ".json"),
        )

        # with open(os.path.join(path, "priors_"+model_name+".json"), "w") as file:
        #     json.dump(simulation_params['priors'][model_name], file, sort_keys=True, indent=4, separators=(",", ": "))
        #   for key in simulation_params['priors'].keys()
        # shutil.rmtree(os.path.join(path))
    plot_priors(simulation_params, os.path.join(output_path, 'figures', 'priors'))
    simulation_params["prior_data_set"] = prior_data_set
    print('==================')
    print('Prior Fitting Simulations Complete')
    print('==================')
    return simulation_params


def calculate_bayes_factor(simulation_params, runfile_path, output_path, n_processes):
    mbar_params = parse_input_yaml(os.path.join(runfile_path, "basic_run.yaml"))
    mbar_params['priors'] = simulation_params['priors']
    compound = simulation_params['compound']
    # Load the data.
    initial_data_set, property_types = prepare_data(mbar_params, compound, scenario='prior')
    print(property_types)
    data_set = choose_test_datapoints(os.path.join(runfile_path, "basic_run.yaml"), simulation_params, initial_data_set,
                                      scenario='main')

    _, property_types = prepare_data(mbar_params, compound, scenario='main')
    print(property_types)
    for property in property_types:
        print(property)
        print(data_set.get_data(property))
    # Build the models.
    models = [
        get_2clj_model(model_name, data_set, property_types, mbar_params)
        for model_name in ["UA", "AUA", "AUA+Q"]
    ]
    print(property_types[0], data_set.get_data(property_types[0]))
    # Fit multivariate normal distributions to this models posterior.
    # This will be used as the analytical reference distribution.
    fitting_directory = os.path.join(output_path, "intermediate", "mbar_fitting")
    os.makedirs(fitting_directory, exist_ok=True)

    initial_parameters = simulation_params['initial_parameters']
    for key in initial_parameters.keys():
        initial_parameters[key] = numpy.asarray(initial_parameters[key])
    with Pool(n_processes) as pool:
        all_fits = pool.map(
            functools.partial(
                fit_to_trace,
                steps=mbar_params["mbar_fit_steps"],
                output_directory=fitting_directory,
                initial_parameters=initial_parameters,
            ),
            models,
        )
    print('==================')
    print('MBAR Fitting Simulations Complete')
    print('==================')
    # Run the MBAR calculations
    results = {}
    mbar_output_path = os.path.join(output_path, 'mbar_results')
    for model, fits in zip(models, all_fits):
        _, reference_model = fits

        # Run the MBAR simulation
        lambda_values = numpy.linspace(0.0, 1.0, 3)

        output_directory = os.path.join(mbar_output_path, f"mbar_{model.name}")

        simulation = MBARIntegration(
            lambda_values=lambda_values,
            model=model,
            steps=mbar_params["mbar_steps"],
            output_directory_path=output_directory,
            reference_model=reference_model,
        )

        _, integral, error = simulation.run(
            reference_model.sample_priors(), number_of_processes=n_processes
        )

        results[model.name] = {"integral": integral, "error": error}
        print('==================')
        print(f'{model.name} complete, integral of {integral} with {error} error')
        print('==================')
    models = []
    evidences = []
    errors = []

    for key in results.keys():
        evidences.append(results[key]['integral'])
        models.append(key)
        errors.append(results[key]['error'])
    output = bayes_factor_from_evidence(models, evidences, errors, output_path + '/figures', plot=True)

    with open(output_path + '/' + f"mbar_{compound}_results.json", "w") as file:
        json.dump(results, file, sort_keys=True, indent=4, separators=(",", ": "))
    with open(output_path + '/' + f"mbar_{compound}_bayes_factors.json", "w") as file:
        json.dump(output, file, sort_keys=True, indent=4, separators=(",", ": "))
    # shutil.rmtree(fitting_directory)
    simulation_params["production_data_set"] = data_set
    print('==================')
    print('MBAR Calculations Complete')
    print('==================')
    return output, results


def rjmc_validation(simulation_params, results, runfile_path, output_path, n_processes):
    # Load in the simulation parameters.
    rjmc_params = parse_input_yaml(os.path.join(runfile_path, 'basic_run.yaml'))
    rjmc_params['priors'] = simulation_params['priors']
    # Load the data.
    _, property_types = prepare_data(rjmc_params, scenario='main')
    data_set = simulation_params['production_data_set']
    print(property_types[0], data_set.get_data(property_types[0]))
    # Build the model / models.
    sub_models = [
        get_2clj_model("AUA", data_set, property_types, rjmc_params),
        get_2clj_model("AUA+Q", data_set, property_types, rjmc_params),
        get_2clj_model("UA", data_set, property_types, rjmc_params),
    ]

    initial_parameters = simulation_params['initial_parameters']
    for key in initial_parameters.keys():
        initial_parameters[key] = numpy.asarray(initial_parameters[key])
    # Load the mapping distributions
    mapping_distributions = []
    maximum_a_posteriori = []
    fitting_directory = os.path.join(output_path, 'intermediate', 'rjmc_fitting')
    os.makedirs(fitting_directory, exist_ok=True)
    with Pool(n_processes) as pool:

        pool.map(
            functools.partial(
                fit_to_trace,
                steps=rjmc_params["rjmc_fit_steps"],
                output_directory=fitting_directory,
                initial_parameters=initial_parameters,
            ),
            sub_models,
        )
    print('==================')
    print('RJMC Fitting Simulations Complete')
    print('==================')
    for model in sub_models:

        fit_path = os.path.join(fitting_directory, f"{model.name}_univariate_fit.json")
        with open(fit_path) as file:
            fit_distributions = json.load(file)

        fit_model = UnconditionedModel(model.name, fit_distributions, {})
        mapping_distributions.append(fit_model.priors)

        # Determine the maximum a posteriori of the fit
        map_parameters = []

        for distribution in fit_model.priors:

            if isinstance(distribution, (distributions.Normal)):
                if isinstance(distribution, (distributions.HalfNormal)):
                    map_parameters.append(0)
                else:
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

    output_directory_path = os.path.join(output_path, 'RJMC_results')
    os.makedirs(output_directory_path, exist_ok=True)

    simulation = BiasedRJMCSimulation(
        model_collection=model_collection,
        initial_parameters=initial_parameters,
        initial_model_index=initial_model_index,
        swap_frequency=rjmc_params["swap_freq"],
        log_biases=bias_factors,
    )

    trace, log_p_trace, percent_deviation_trace = simulation.run(
        warm_up_steps=int(rjmc_params["rjmc_fit_steps"] * 0.1),
        steps=rjmc_params["rjmc_steps"],
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
    # shutil.rmtree(fitting_directory)
    print('==================')
    print('RJMC Simulations Complete')
    print('==================')
    return rjmc_counts


def mcmc_placeholder(simulation_params):
    # Load the data.
    data_set, property_types = prepare_data(simulation_params, scenario='main')

    # Set some initial parameter close to the MAP taken from
    # other runs of C2H6.
    initial_parameters = simulation_params['initial_parameters']
    model = get_2clj_model("AUA+Q", data_set, property_types, simulation_params)
    # Run the simulation.
    simulation = MCMCSimulation(
        model_collection=model, initial_parameters=numpy.asarray(initial_parameters[model.name])
    )

    trace, log_p_trace, percent_deviation_trace = simulation.run(
        warm_up_steps=int(simulation_params["steps"] * 0.2),
        steps=simulation_params["steps"],
    )
    model.plot(trace, log_p_trace, percent_deviation_trace, show=True)


def mcmc_benchmarking(simulation_params, runfile_path, output_path):
    # Load the data.
    _, property_types = prepare_data(simulation_params, scenario='main')
    # Set some initial parameter close to the MAP taken from
    # other runs of C2H6.
    initial_parameters = simulation_params['initial_parameters']

    # Build the model / models.
    model_elpds = {}
    model_deviations = {}
    model_keys = ["UA", "AUA", "AUA+Q"]
    for model_name in model_keys:

        data_set = copy.deepcopy(simulation_params['production_data_set'])
        model = get_2clj_model(model_name, data_set, property_types, simulation_params)

        # Run the simulation.
        simulation = MCMCSimulation(
            model_collection=model, initial_parameters=numpy.asarray(initial_parameters[model.name])
        )
        path = os.path.join(output_path, 'intermediate', 'mcmc_benchmarking')
        os.makedirs(path, exist_ok=True)

        trace, log_p_trace, percent_deviation_trace = simulation.run(
            warm_up_steps=int(simulation_params["mcmc_steps"] * 0.2),
            steps=simulation_params["mcmc_steps"],
            output_directory=path
        )

        params = simulation.fit_prior_exponential()
        # calculate summary statistics
        benchmark_filepath = os.path.join(runfile_path, 'basic_run.yaml')
        data_set.concatenate_datasets(simulation_params['prior_data_set'])
        test_params = parse_input_yaml(benchmark_filepath)
        test_set = choose_test_datapoints(benchmark_filepath, simulation_params, data_set)
        samples = sample_trace(trace, test_set, model_name, test_params["number_samples"])

        test_model = get_2clj_model(model_name, test_set, property_types, simulation_params)
        elpd, elppd = calculate_elpd(test_model, property_types, samples)
        # model.plot(trace, log_p_trace, percent_deviation_trace, show=True)
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
        data_set = None
    # shutil.rmtree(path)
    plot_deviations(model_deviations, model_keys, property_types, output_path)
    for property in property_types:
        print(property)
        for key in model_elpds.keys():
            print(f"For model {key},"
                  f" -ELPPD is {model_elpds[key][1][property]}"
                  f" tested against {len(model_elpds[key][0][property])}"
                  f" data points (average stdev away from experiment "
                  f"= {numpy.sqrt(2 * (model_elpds[key][1][property] / len(model_elpds[key][0][property])))})")
            if property == NISTDataType.LiquidDensity:
                property_name = 'Density'
            elif property == NISTDataType.SaturationPressure:
                property_name = 'Saturation Pressure'
            elif property == NISTDataType.SurfaceTension:
                property_name = 'Surface Tension'
            model_elpds[key][0][property_name] = model_elpds[key][0][property]
            model_elpds[key][1][property_name] = model_elpds[key][1][property]
            del model_elpds[key][0][property], model_elpds[key][1][property]
        print('==================')
        print('Benchmarking Simulations Complete')
        print('==================')
    with open(output_path + '/' + f"ELPPD.json", "w") as file:
        json.dump(model_elpds, file, sort_keys=True, indent=4, separators=(",", ": "))


def bayes_factor_from_evidence(models, evidences, errors, filepath, plot=False):
    output = {}
    output['models'] = copy.deepcopy(models)
    evidences = numpy.asarray(evidences)
    errors = numpy.asarray(errors)
    maxarg = numpy.argmax(evidences)
    subtract_values = evidences - evidences[maxarg]
    output['log Bayes factor'] = subtract_values.tolist()
    subtract_variance = numpy.sqrt(errors ** 2 + errors[maxarg] ** 2)
    subtract_variance[maxarg] = 0
    output['log Bayes factor uncertainty'] = numpy.sqrt(subtract_variance).tolist()
    exp_values = numpy.exp(subtract_values)

    exp_variance = numpy.multiply(exp_values, numpy.sqrt(subtract_variance)) ** 2

    divide_values = exp_values / exp_values[maxarg]

    divide_variance = numpy.multiply(
        (exp_variance / exp_values ** 2 + (exp_variance[maxarg] / exp_values[maxarg] ** 2)),
        divide_values ** 2)
    divide_stdev = numpy.sqrt(divide_variance)
    output['Bayes factor'] = divide_values.tolist()
    output['Bayes factor uncertainty'] = divide_stdev.tolist()
    if plot is True:
        ref_model = models[numpy.argmax(subtract_values)]
        subtract_stdev = numpy.sqrt(numpy.delete(subtract_variance, numpy.argmax(subtract_values)))
        subtract_values = numpy.delete(subtract_values, numpy.argmax(subtract_values))
        models.pop(numpy.argmax(subtract_values))
        chart_models = []
        for model in models:
            chart_models.append(model + '/' + ref_model)
        fig, ax1 = plt.subplots(figsize=(8, 8), constrained_layout=True)
        plt.title(f'Model Evidences in favor of {ref_model}', fontsize=24)
        ax1.bar(chart_models, subtract_values, yerr=subtract_stdev, color=['r', 'orange'])
        ax1.set_ylabel('ln(Bayes Factor) vs ' + ref_model, fontsize=16)
        ax1.axhline(-1, color='k', ls='--', label='Significant Evidence for ' + ref_model, lw=1)
        ax1.axhline(-3, color='k', ls='--', label='Strong Evidence for ' + ref_model, lw=1)
        ax1.axhline(-5, color='k', ls='--', label='Very Strong Evidence for ' + ref_model, lw=1)

        # ax1.set_xlabel('Model', fontsize=16)

        # ax1.legend(fontsize=12)
        pos = [-1, -3, -5]
        evidence_label = ['Significant \n Evidence', 'Strong \n Evidence', 'Very Strong \n Evidence']
        ax1.tick_params(axis='y', labelsize=16)
        ax1.tick_params(axis='x', labelsize=24)
        ax1.set_ylabel('ln(Bayes Factor) vs ' + ref_model, fontsize=22)
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.axes.get_ylim())
        ax2.tick_params(labelsize=18)
        ax2.set_yticks(pos)
        ax2.set_yticklabels(evidence_label, rotation=-90, va='center')
        plt.savefig(filepath + '/log_bayes_factors.png')

    return output


def plot_priors(simulation_params, path):
    model_names = ['UA', 'AUA', 'AUA+Q']
    parameter_names = ['epsilon', 'sigma', 'L', 'Q']
    dist_expon = dist.expon
    dist_norm = dist.norm
    dist_gamma = dist.gamma

    for param_name in parameter_names:
        dists = []
        models = []
        for model_name in model_names:
            if param_name in simulation_params['priors'][model_name]:
                models.append(model_name)
                if simulation_params['priors'][model_name][param_name][0] == 'exponential':
                    dists.append(dist_expon(*simulation_params['priors'][model_name][param_name][1]))
                elif simulation_params['priors'][model_name][param_name][0] == 'normal':
                    dists.append(dist_norm(*simulation_params['priors'][model_name][param_name][1]))
                elif simulation_params['priors'][model_name][param_name][0] == 'gamma':
                    dists.append(dist_gamma(simulation_params['priors'][model_name][param_name][1][0],
                                            scale=simulation_params['priors'][model_name][param_name][1][1]))
                elif simulation_params['priors'][model_name][param_name][0] == 'uniform':
                    dists.append(dist.uniform(loc=simulation_params['priors'][model_name][param_name][1][0],
                                              scale=simulation_params['priors'][model_name][param_name][1][1]))
        mins = []
        maxs = []
        for distribution in dists:
            mins.append(distribution.ppf(0.01))
            maxs.append(distribution.ppf(0.99))
        x_vec = numpy.linspace(min(mins), max(maxs), num=500)
        plt.cla()
        for distribution, model in zip(dists, models):
            plt.plot(x_vec, distribution.pdf(x_vec), label=model)
        plt.legend()
        plt.savefig(os.path.join(path, param_name + '_all.png'))

    return


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
    output_location = '/home/owenmadin/storage/LINCOLN1/rjmc_output'
    args = parser.parse_args()
    main(args.compound, args.properties, output_location)

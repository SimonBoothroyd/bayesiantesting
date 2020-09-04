import numpy
import yaml
import pandas
import os

from bayesiantesting import unit
from bayesiantesting.datasets.nist import NISTDataSet, NISTDataType
from bayesiantesting.surrogates import StollWerthSurrogate
from studies.utilities import parse_input_yaml, prepare_data
import matplotlib.pyplot as plt


def sample_trace(trace, test_set, model_name, n_samples):
    samples = trace[numpy.random.choice(trace.shape[0], n_samples, replace=False), 1:]
    return samples


def choose_test_datapoints(filepath, simulation_params, train_set):
    test_params = parse_input_yaml(filepath)
    simulation_params["number_data_points"] = test_params["number_data_points"]
    simulation_params["trange"] = test_params["trange"]

    test_set, property_types = prepare_data(simulation_params, filtering=False)

    test_set.remove_datapoints(train_set)
    minimum_temperature = (
            simulation_params["trange"][0]
            * test_set.critical_temperature.value.to(unit.kelvin).magnitude
    )
    maximum_temperature = (
            simulation_params["trange"][1]
            * test_set.critical_temperature.value.to(unit.kelvin).magnitude
    )

    test_set.filter(
        minimum_temperature * unit.kelvin,
        maximum_temperature * unit.kelvin,
        simulation_params["number_data_points"],
    )

    return test_set


def calculate_deviations(property_types, test_set, train_set, samples):
    surrogate = StollWerthSurrogate(test_set.molecular_weight)
    test_results = {}
    for property_type in property_types:
        property_measurements = []
        for sample in samples:
            values = surrogate.evaluate(property_type, sample, numpy.asarray(test_set.get_data(property_type)['T (K)']))
            property_measurements.append(values)
        property_measurements = numpy.asarray(property_measurements).transpose()

        test_results[property_type] = {}
        test_results[property_type]['mean'] = property_measurements.mean(axis=1)
        test_results[property_type]['std'] = property_measurements.std(axis=1)
        test_results[property_type]['temperature'] = numpy.asarray(test_set.get_data(property_type)['T (K)'])
        test_results[property_type]['measurement uncertainties'] = numpy.asarray(
            test_set.get_data(property_type)[test_set.get_data(property_type).columns[2]])
        test_results[property_type]['measurements'] = numpy.asarray(
            test_set.get_data(property_type)[test_set.get_data(property_type).columns[1]])

    train_results = {}
    for property_type in property_types:
        property_measurements = []
        for sample in samples:
            values = surrogate.evaluate(property_type, sample,
                                        numpy.asarray(train_set.get_data(property_type)['T (K)']))
            property_measurements.append(values)
        property_measurements = numpy.asarray(property_measurements).transpose()
        train_results[property_type] = {}
        train_results[property_type]['mean'] = property_measurements.mean(axis=1)
        train_results[property_type]['std'] = property_measurements.std(axis=1)
        train_results[property_type]['temperature'] = numpy.asarray(train_set.get_data(property_type)['T (K)'])
        train_results[property_type]['measurement uncertainties'] = numpy.asarray(
            train_set.get_data(property_type)[train_set.get_data(property_type).columns[2]])
        train_results[property_type]['measurements'] = numpy.asarray(
            train_set.get_data(property_type)[train_set.get_data(property_type).columns[1]])

    return test_results, train_results


def plot_deviations(model_deviations, model_names, property_types, path):
    for property in property_types:

        xlabel = 'T (K)'
        if property == NISTDataType.LiquidDensity:
            title = 'Density % deviation'
            ylabel = r'$\rho_l$ (kg/$\mathrm{m}^3)$'
        elif property == NISTDataType.SaturationPressure:
            title = ' Saturation pressure % deviation'
            ylabel = r'$P_{sat}$ (kPa)'
        elif property == NISTDataType.SurfaceTension:
            title = 'Surface tension % deviation'
            ylabel = r'$\gamma$ (N/m)'
        plt.figure(figsize=(8, 8))
        # plot data points
        plt.xlabel(xlabel, fontsize=16)
        # plt.ylabel(ylabel, fontsize=16)
        plt.title(title, fontsize=24)
        # plot test
        markers = ['s', 'o', 'v']
        colors = ['red', 'blue', 'orange']
        plt.axhline(0, color='grey', linestyle='--', label='Experiment')
        for model_name, marker, color in zip(model_names, markers, colors):
            plt.scatter(model_deviations[model_name][0][property]['temperature'],
                        100 * (model_deviations[model_name][0][property]['mean']
                               - model_deviations[model_name][0][property]['measurements']) /
                        model_deviations[model_name][0][property]['measurements'],
                        marker=marker,
                        facecolors='none',
                        edgecolors=color,
                        label=model_name + ' Test Deviations')
            # plot train
            plt.scatter(model_deviations[model_name][1][property]['temperature'],
                        100 * (model_deviations[model_name][1][property]['mean']
                               - model_deviations[model_name][1][property]['measurements']) /
                        model_deviations[model_name][1][property]['measurements'],
                        marker=marker,
                        facecolors=color,
                        edgecolors=color,
                        label=model_name + ' Train Deviations')
        plt.legend()
        figpath = os.path.join(path, 'figures', 'benchmarking')
        os.makedirs(figpath, exist_ok=True)
        plt.savefig(figpath + '/' + title + '.png')
        plt.clf()


def plot_points(model_deviations, model_names, property_types):
    for property in property_types:

        xlabel = 'T (K)'
        if property == NISTDataType.LiquidDensity:
            title = r'Density benchmark'
            ylabel = r'$\rho_l$ (kg/$\mathrm{m}^3)$'
        elif property == NISTDataType.SaturationPressure:
            title = r' Saturation pressure benchmark'
            ylabel = r'$P_{sat}$ (kPa)'
        elif property == NISTDataType.SurfaceTension:
            title = r'Surface tension benchmark'
            ylabel = r'$\gamma$ (N/m)'
        plt.figure(figsize=(8, 8))
        # plot data points
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.title(title, fontsize=24)
        # plot test
        plt.scatter(model_deviations[model_names[0]][0][property]['temperature'],
                    model_deviations[model_names[0]][0][property]['measurements'],
                    marker='o',
                    facecolors='none',
                    edgecolors='grey',
                    label='Measured Test Points')
        # plot train
        plt.scatter(model_deviations[model_names[0]][1][property]['temperature'],
                    model_deviations[model_names[0]][1][property]['measurements'],
                    marker='o',
                    facecolors='grey',
                    edgecolors='grey',
                    label='Measured Train Points')
        colors = ['red', 'blue', 'orange']
        markers = ['s', 'o', 'v']
        for model_name, color, marker in zip(model_names, colors, markers):
            # plot per model deviations
            plt.scatter(model_deviations[model_name][0][property]['temperature'],
                        model_deviations[model_name][0][property]['mean'],
                        marker=marker,
                        facecolors='none',
                        edgecolors=color,
                        label=model_name + ' Test Points')
            plt.scatter(model_deviations[model_name][1][property]['temperature'],
                        model_deviations[model_name][1][property]['mean'],
                        marker=marker,
                        facecolors=color,
                        edgecolors=color,
                        label=model_name + ' Train Points')
        plt.legend()
        plt.show()


def calculate_elpd(model, property_types, samples):
    elpd_dict = {}
    for property in property_types:
        elpd_dict[property] = []
    for sample in samples:
        logs_sample = model.evaluate_pointwise_log_likelihood(sample)
        for key in logs_sample.keys():
            elpd_dict[key].append(logs_sample[key])


    if property == NISTDataType.LiquidDensity:
        property_name = 'Density'
    elif property == NISTDataType.SaturationPressure:
        property_name = 'Saturation Pressure'
    elif property == NISTDataType.SurfaceTension:
        property_name = 'Surface Tension'
    elppd_dict = {}
    for key, value in elpd_dict.items():
        elpd_dict[key] = numpy.asarray(value).transpose().mean(1)
        elppd_dict[key] = numpy.sum(elpd_dict[key])
        elpd_dict[key] = elpd_dict[key].tolist()
    return elpd_dict, elppd_dict


def clean_test_data(test_set, train_set):
    # removes train set values from a test set and removes values at duplicate temperatures (keeps last)

    df = pandas.merge(test_set, train_set, how='outer', indicator=True) \
        .query("_merge != 'both'") \
        .drop('_merge', axis=1) \
        .reset_index(drop=True)
    df = df.drop_duplicates(subset='T (K)', keep='last')
    return df


def calculate_rmses(results):
    rmses = []
    for i in range(len(results)):
        for j in range(len(results[i][2])):
            rmse = numpy.sqrt(
                numpy.sum((numpy.asarray(results[i][1][results[i][1].columns[1]]) - results[i][2][j][1]) ** 2) / len(
                    results[i][2][j][1]))
            results[i][2][j].append(rmse)
        rmses.append(numpy.asarray(results[i][2])[:, 2])
    rmses = numpy.asarray(rmses)
    return results, rmses

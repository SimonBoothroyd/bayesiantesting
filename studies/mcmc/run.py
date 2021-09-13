#!/usr/bin/env python3

import numpy

from bayesiantesting.kernels import MCMCSimulation
from studies.utilities import get_2clj_model, parse_input_yaml, prepare_data
from studies.benchmarking import calculate_elpd, sample_trace, choose_test_datapoints, calculate_deviations, \
    plot_deviations, plot_points


def main():

    simulation_params = parse_input_yaml("basic_run.yaml")

    # Load the data.
    data_set, property_types = prepare_data(simulation_params)

    # Set some initial parameter close to the MAP taken from
    # other runs of C2H6.
    initial_parameters = {
        "UA": numpy.array([250, 0.35]),
        "AUA": numpy.array([250.0, 0.35, 0.22]),
        "AUA+Q": numpy.array([250.0, 0.35, 0.22, 0.5]),
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

        trace, log_p_trace, percent_deviation_trace = simulation.run(
            warm_up_steps=int(simulation_params["steps"] * 0.2),
            steps=simulation_params["steps"],
        )

        params = simulation.fit_prior_exponential()
        print(params)



        # calculate summary statistics
        benchmark_filepath = 'evaluate_params.yaml'
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

    for property in property_types:
        print(property)
        for key in model_elpds.keys():
            print(f"For model {key},"
                  f" -ELPPD is {model_elpds[key][1][property]}"
                  f" tested against {len(model_elpds[key][0][property])} data points (average stdev away from experiment "
                  f"= {numpy.sqrt(2 * (model_elpds[key][1][property]/len(model_elpds[key][0][property])))}")
        print('==================')
    plot_deviations(model_deviations, model_keys, property_types, path='.')





if __name__ == "__main__":
    main()

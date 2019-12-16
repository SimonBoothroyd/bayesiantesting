#!/usr/bin/env python3
import json
import os
import numpy

from bayesiantesting.kernels.rjmc import BiasedRJMCSimulation
from bayesiantesting.models.continuous import UnconditionedModel
from bayesiantesting.models.discrete import TwoCenterLJModelCollection
from bayesiantesting.utils import distributions
from studies.utilities import get_2clj_model, parse_input_yaml, prepare_data


def main():

    # Load in the simulation parameters.
    simulation_params = parse_input_yaml("basic_run.yaml")

    # Load the data.
    data_set, property_types = prepare_data(simulation_params)

    # Build the model / models.
    sub_models = [
        get_2clj_model("AUA", data_set, property_types, simulation_params),
        get_2clj_model("AUA+Q", data_set, property_types, simulation_params),
        get_2clj_model("UA", data_set, property_types, simulation_params),
    ]

    # Load the mapping distributions
    mapping_distributions = []
    maximum_a_posteriori = []

    for model in sub_models:

        fit_path = os.path.join(
            simulation_params["compound"], f"{model.name}_univariate_fit.json"
        )

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
    bias_file_name = f"mbar_{simulation_params['compound']}_results.json"

    with open(bias_file_name) as file:
        bias_factor_dictionary = json.load(file)

    bias_factors = [
        bias_factor_dictionary[model.name]["integral"] for model in sub_models
    ]
    bias_factors = -numpy.asarray(bias_factors)

    # Draw the initial parameter values from the model priors.
    initial_model_index = 1  # torch.randint(len(sub_models), (1,)).item()

    # initial_parameters = generate_initial_parameters(sub_models[initial_model_index])
    initial_parameters = maximum_a_posteriori[initial_model_index]

    output_directory_path = simulation_params["compound"]

    simulation = BiasedRJMCSimulation(
        model_collection=model_collection,
        initial_parameters=initial_parameters,
        initial_model_index=initial_model_index,
        swap_frequency=simulation_params["swap_freq"],
        log_biases=bias_factors,
        output_directory_path=output_directory_path,
    )

    simulation.run(
        warm_up_steps=int(simulation_params["steps"] * 0.1),
        steps=simulation_params["steps"],
    )


if __name__ == "__main__":
    main()

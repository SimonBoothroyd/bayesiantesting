import numpy
import yaml
import pandas
import os
import json

from bayesiantesting import unit
from bayesiantesting.datasets.nist import NISTDataSet, NISTDataType
from bayesiantesting.models.continuous import TwoCenterLJModel
from bayesiantesting.surrogates import StollWerthSurrogate
from bayesiantesting.models.continuous import UnconditionedModel
from bayesiantesting.kernels import MCMCSimulation

def parse_input_yaml(filepath):

    print("Loading simulation params from " + filepath + "...")

    with open(filepath) as file:
        simulation_params = yaml.load(file, Loader=yaml.SafeLoader)

    return simulation_params


def prepare_data(simulation_params, compound=None, filtering=True):
    """From input parameters, pull appropriate experimental data and
    uncertainty information.
    """

    if compound is None:
        compound = simulation_params["compound"]

    # Retrieve the constants and thermophysical data
    data_set = NISTDataSet(compound)
    if filtering is True:
        # Filter the data to selected conditions.
        minimum_temperature = (
            simulation_params["trange"][0]
            * data_set.critical_temperature.value.to(unit.kelvin).magnitude
        )
        maximum_temperature = (
            simulation_params["trange"][1]
            * data_set.critical_temperature.value.to(unit.kelvin).magnitude
        )

        data_set.filter(
            minimum_temperature * unit.kelvin,
            maximum_temperature * unit.kelvin,
            simulation_params["number_data_points"],
        )

    property_types = []

    if simulation_params["properties"] == "All":
        property_types.extend(data_set.data_types)
    else:
        if "rhol" in simulation_params["properties"]:
            property_types.append(NISTDataType.LiquidDensity)
        if "Psat" in simulation_params["properties"]:
            property_types.append(NISTDataType.SaturationPressure)

    return data_set, property_types


def get_2clj_model(model_name, data_set, property_types, simulation_params):

    priors = {
        "epsilon": simulation_params["priors"]["epsilon"],
        "sigma": simulation_params["priors"]["sigma"],
    }
    fixed = {}

    if model_name == "AUA":

        priors["L"] = simulation_params["priors"]["L"]
        fixed["Q"] = 0.0

    elif model_name == "AUA+Q":

        priors["L"] = simulation_params["priors"]["L"]
        priors["Q"] = simulation_params["priors"]["Q"]

    elif model_name == "UA":

        fixed["L"] = data_set.bond_length.to(unit.nanometer).magnitude
        fixed["Q"] = 0.0

    else:

        raise NotImplementedError()

    model = TwoCenterLJModel(
        model_name,
        priors,
        fixed,
        data_set,
        property_types,
        StollWerthSurrogate(data_set.molecular_weight),
    )

    return model


def generate_initial_parameters(model):

    initial_log_p = -numpy.inf
    initial_parameters = None

    counter = 0

    while numpy.isinf(initial_log_p) and counter < 10000:

        initial_parameters = model.sample_priors()
        initial_log_p = model.evaluate_log_posterior(initial_parameters)

        counter += 1

    initial_parameters = model.find_maximum_a_posteriori(initial_parameters)
    initial_log_p = model.evaluate_log_posterior(initial_parameters)

    if numpy.isnan(initial_log_p) or numpy.isinf(initial_log_p):

        raise ValueError(
            "The initial values could not be set without yielding "
            "a NaN / inf log posterior"
        )

    return initial_parameters


def prior_dictionary_to_json(prior_settings, file_path):
    """Saves a dictionary of prior settings to JSON.

    Parameters
    ----------
    prior_settings: dict
        The dictionary of settings.
    file_path: str
        The file path to save the JSON to.
    """

    json_dictionary = {}

    for label in prior_settings:

        json_label = label

        if isinstance(json_label, tuple):
            json_label = ",".join(json_label)

        json_dictionary[json_label] = [prior_settings[label][0], []]

        for parameter in prior_settings[label][1]:

            parameter_list = parameter.tolist()
            json_dictionary[json_label][1].append(parameter_list)

    with open(file_path, "w") as file:
        json.dump(
            json_dictionary, file, sort_keys=False, indent=4, separators=(",", ": ")
        )


def prior_dictionary_from_json(file_path):
    """Loads a dictionary of prior settings from a JSON
    file.

    Parameters
    ----------
    file_path: str
        The file path to load the JSON from.

    Returns
    -------
    dict
        The dictionary of prior settings.
    """

    with open(file_path, "r") as file:
        json_dictionary = json.load(file)

    prior_settings = {}

    for json_label in json_dictionary:

        label = json_label

        if label.find(",") >= 0:
            label = tuple(label.split(","))

        prior_settings[label] = [json_dictionary[json_label][0], []]

        for parameter_list in json_dictionary[json_label][1]:

            parameter = parameter_list

            if isinstance(parameter, list):
                parameter = numpy.asarray(parameter_list)

            prior_settings[label][1].append(parameter)

    return prior_settings


def fit_prior_to_trace(parameter_trace):
    """Fits either a normal or a half normal distribution
    to a given trace.

    Parameters
    ----------
    parameter_trace: numpy.ndarray
        The parameter trace from an `MCMCSimulation` simulation
        with shape=(n_steps).

    Returns
    -------
    list
        The prior settings.
    """
    loc = numpy.mean(parameter_trace)
    scale = numpy.std(parameter_trace)

    if loc - 5.0 * scale > 0.0:
        # Check to make sure whether a half-normal distribution may be
        # more appropriate.
        return ["normal", [loc, scale]]

    scale = numpy.sqrt(numpy.sum(parameter_trace ** 2) / len(parameter_trace))
    return ["half normal", [scale]]


def fit_to_trace(model, output_directory, initial_parameters, warm_up_steps, use_existing=True):
    """Fits a multivariate gaussian distribution to the posterior
    of the model as sampled by an MCMC simulation.

    Parameters
    ----------
    model: Model
        The model to sample.
    output_directory: str
        The directory to store the working files in.
    initial_parameters: numpy.ndarray
        The parameters to start the simulation from.
    use_existing: bool
        If True, any existing fits will be used rather than regenerating
        new fits.

    Returns
    -------
    UnconditionedModel
        The fitted univariate model.
    UnconditionedModel
        The fitted multivariate model.
    """

    trace_path = os.path.join(output_directory, model.name, "trace.npy")

    if not use_existing or not os.path.isfile(trace_path):
        # initial_parameters = generate_initial_parameters(model)
        initial_parameters = initial_parameters[model.name]

        # Run a short MCMC simulation to get better initial parameters
        simulation = MCMCSimulation(
            model_collection=model, initial_parameters=initial_parameters,
        )

        simulation.run(
            warm_up_steps/3, steps=warm_up_steps, output_directory=output_directory
        )

    trace = numpy.load(trace_path)

    # Fit the univariate distributions.
    univariate_prior_dictionary = {}

    for index, label in enumerate(model.trainable_parameter_labels):

        parameter_trace = trace[:, index + 1]
        univariate_prior_dictionary[label] = fit_prior_to_trace(parameter_trace)

    prior_dictionary_to_json(
        univariate_prior_dictionary,
        os.path.join(output_directory, f"{model.name}_univariate_fit.json"),
    )

    # Fit the multivariate distribution.
    n_multivariate_parameters = model.n_trainable_parameters

    if model.name == "AUA+Q":
        n_multivariate_parameters -= 1

    multivariate_mean = numpy.mean(trace[:, 1 : 1 + n_multivariate_parameters], axis=0)
    multivariate_covariance = numpy.cov(trace[:, 1 : 1 + n_multivariate_parameters].T)

    multivariate_key = tuple(
        [
            label
            for label in model.trainable_parameter_labels[:n_multivariate_parameters]
        ]
    )

    multivariate_prior_dictionary = {
        multivariate_key: [
            "multivariate normal",
            [multivariate_mean, multivariate_covariance],
        ]
    }

    for index, label in enumerate(
        model.trainable_parameter_labels[n_multivariate_parameters:]
    ):

        parameter_trace = trace[:, n_multivariate_parameters + index + 1]
        multivariate_prior_dictionary[label] = fit_prior_to_trace(parameter_trace)

    prior_dictionary_to_json(
        multivariate_prior_dictionary,
        os.path.join(output_directory, f"{model.name}_multivariate_fit.json"),
    )

    return (
        UnconditionedModel(f"{model.name}_univariate", univariate_prior_dictionary, {}),
        UnconditionedModel(
            f"{model.name}_multivariate", multivariate_prior_dictionary, {}
        ),
    )
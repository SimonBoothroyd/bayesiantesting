import os
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from bayesiantesting import unit
from bayesiantesting.utils import get_data_filename


class NISTDataType(Enum):

    LiquidDensity = "Liquid Density"
    SaturationPressure = "Saturation Pressure"
    SurfaceTension = "Surface Tension"


class NISTDataSet:
    """A wrapper around data obtained from the NIST Thermodynamics
    Research Center (TRC) for a given compound.
    """

    @property
    def compound(self) -> str:
        """str: The compound for which this data was collected."""
        return self._compound

    @property
    def data_types(self) -> List[NISTDataType]:
        """list of NISTDataType: The types of data stored in this set."""
        return [
            NISTDataType.LiquidDensity,
            NISTDataType.SaturationPressure,
            NISTDataType.SurfaceTension,
        ]

    @property
    def critical_temperature(self) -> unit.Measurement:
        """unit.Measurement: The critical temperature of the compound."""
        return self._critical_temperature

    @property
    def molecular_weight(self) -> unit.Quantity:
        """unit.Quantity: The molecule weight of the compound."""
        return self._molecular_weight

    @property
    def bond_length(self) -> unit.Quantity:
        """unit.Quantity: The bond length of the compound."""
        return self._bond_length

    def __init__(self, compound: str):
        """Constructs a new `NISTDataSet` object.

        Parameters
        ----------
        compound: str
            The compound to load the data for.
        """
        self._compound = compound

        self._bond_length = 0.0

        (
            self._critical_temperature,
            self._molecular_weight,
            self._bond_length,
            self._data,
            self._precisions,
        ) = self._load_data()

    def _load_data(
        self,
    ) -> Tuple[
        unit.Measurement,
        unit.Quantity,
        unit.Quantity,
        Dict[NISTDataType, pd.DataFrame],
        Dict[NISTDataType, np.ndarray],
    ]:
        """Loads the data for a chosen compound from the data directory.

        Returns
        -------
        unit.Measurement:
            The critical temperature.
        unit.Quantity:
            The molecular weight.
        unit.Quantity:
            The bond length.
        dict of NISTDataType and pandas.DataFrame:
            A dictionary of the different data points.
        dict of NISTDataType and numpy.ndarray
            The precision in each of the data types.
        """

        raw_critical_temperature = np.loadtxt(
            get_data_filename(os.path.join("trc_data", self._compound, "Tc.txt")),
            skiprows=1,
        )

        critical_temperature = (raw_critical_temperature[0] * unit.kelvin).plus_minus(
            raw_critical_temperature[1]
        )

        raw_molecular_weight = np.loadtxt(
            get_data_filename(os.path.join("trc_data", self._compound, "Mw.txt")),
            skiprows=1,
        )

        molecular_weight = raw_molecular_weight * unit.gram / unit.mole

        bond_lengths = pd.read_csv(
            get_data_filename("nist_bond_lengths.txt"), delimiter="\t"
        )
        bond_length = (
            bond_lengths[bond_lengths.Compound == self._compound].values[0][1]
            * unit.angstrom
        )

        data = {}
        precisions = {}

        file_names = {
            NISTDataType.LiquidDensity: "rhoL",
            NISTDataType.SaturationPressure: "Pv",
            NISTDataType.SurfaceTension: "SurfTens",
        }

        for data_type, file_name in file_names.items():

            data_frame = pd.read_csv(
                get_data_filename(
                    os.path.join("trc_data", self._compound, f"{file_name}.txt")
                ),
                sep="\t",
            )

            data_frame = data_frame.dropna()
            data[data_type] = data_frame

            precisions[data_type] = self._calculate_precision(
                data_frame, data_type, critical_temperature
            )

        return critical_temperature, molecular_weight, bond_length, data, precisions

    @staticmethod
    def _calculate_precision(
        data_frame: pd.DataFrame,
        data_type: NISTDataType,
        critical_temperature: unit.Measurement,
    ) -> np.ndarray:

        critical_temperature_kelvin = critical_temperature.value.to(
            unit.kelvin
        ).magnitude

        # Extract data from our data arrays
        data = np.asarray(data_frame)

        temperatures = data[:, 0]
        values = data[:, 1]
        experimental_uncertainties = data[:, 2]

        correllation_uncertainties = (
            NISTDataSet._evaluate_uncertainty_model(
                temperatures, critical_temperature_kelvin, data_type
            )
            * values
        )
        total_uncertainties = np.sqrt(
            correllation_uncertainties ** 2 + experimental_uncertainties ** 2
        )

        # Calculate the estimated standard deviation
        standard_deviation = total_uncertainties / 2.0

        # Calculate the precision in each property
        precision = np.sqrt(1.0 / standard_deviation)
        return precision

    @staticmethod
    def _evaluate_uncertainty_model(
        temperatures: np.ndarray,
        critical_temperature: float,
        property_type: NISTDataType,
    ) -> np.ndarray:
        """Evaluates a linear models for uncertainties in the 2CLJQ correlation we are
        using, determined from Messerly analysis of figure from Stobener, Stoll, Werth.

        Parameters
        ----------
        temperatures: numpy.ndarray
            The temperatures to evaluate the model at in kelvin.
        critical_temperature: float
            The critical temperature in kelvin.
        property_type: NISTDataType
            The type of property to evaluate the uncertainties of.

        Returns
        -------
        numpy.ndarray:
            The results of evaluating the uncertainty model.
        """
        temperature_ratio = temperatures / critical_temperature
        uncertainties = np.zeros(np.size(temperature_ratio))

        if property_type == NISTDataType.LiquidDensity:

            # Starts at 0.3% for low values and ramps up to 1% for large values
            for i in range(np.size(temperature_ratio)):

                if temperature_ratio[i] < 0.9:
                    uncertainties[i] = 0.3
                elif 0.9 <= temperature_ratio[i] <= 0.95:
                    uncertainties[i] = 0.3 + (1 - 0.3) * (
                        temperature_ratio[i] - 0.9
                    ) / (0.95 - 0.9)
                else:
                    uncertainties[i] = 1.0

        elif property_type == NISTDataType.SaturationPressure:

            # Starts at 20% for low values and ramps down to 2% for large values
            for i in range(np.size(temperature_ratio)):

                if temperature_ratio[i] <= 0.55:
                    uncertainties[i] = 20
                elif 0.55 <= temperature_ratio[i] <= 0.7:
                    uncertainties[i] = 20 + (2 - 20) * (temperature_ratio[i] - 0.55) / (
                        0.7 - 0.55
                    )
                else:
                    uncertainties[i] = 2.0

        elif property_type == NISTDataType.SurfaceTension:

            # Starts at 4% for low values and ramps up to 12% for higher values
            for i in range(np.size(temperature_ratio)):

                if temperature_ratio[i] <= 0.75:
                    uncertainties[i] = 4
                elif 0.75 <= temperature_ratio[i] <= 0.95:
                    uncertainties[i] = 4 + (12 - 4) * (temperature_ratio[i] - 0.75) / (
                        0.95 - 0.75
                    )
                else:
                    uncertainties[i] = 12.0

        else:
            raise NotImplementedError()

        uncertainties /= 100
        return uncertainties

    def filter(
        self,
        minimum_temperature: unit.Quantity,
        maximum_temperature: unit.Quantity,
        maximum_data_points: int,
    ):
        """Filters a data frame based on a number of specified criteria..

        Parameters
        ----------
        minimum_temperature: unit.Quantity
            All data points below this temperature will be discarded.
        maximum_temperature: unit.Quantity
            All data points above this temperature will be discarded.
        maximum_data_points: int
            The maximum number of data points to load.
        """

        minimum_temperature_kelvin = minimum_temperature.to(unit.kelvin).magnitude
        maximum_temperature_kelvin = maximum_temperature.to(unit.kelvin).magnitude

        for data_type in self.data_types:

            data_frame = self._data[data_type]

            data_frame = data_frame[
                data_frame.values[:, 0] > minimum_temperature_kelvin
            ]
            data_frame = data_frame[
                data_frame.values[:, 0] < maximum_temperature_kelvin
            ]

            if (
                maximum_data_points > 1
                and int(np.floor(data_frame.shape[0] / (maximum_data_points - 1))) == 0
            ):
                slicer = 1
            else:
                slicer = int(
                    np.floor(data_frame.shape[0] / max(1, maximum_data_points - 1))
                )

            self._data[data_type] = data_frame[::slicer]

            self._precisions[data_type] = self._calculate_precision(
                self._data[data_type], data_type, self._critical_temperature
            )

    def get_data(self, data_type: NISTDataType) -> pd.DataFrame:
        """Returns the data of the specified data type.

        Parameters
        ----------
        data_type: NISTDataType

        Returns
        -------
        pandas.DataFrame:
            An data frame with columns of the measured temperature (in kelvin),
            the measured value, and the uncertainty in the measured value.
        """
        return self._data[data_type]

    def get_precision(self, data_type: NISTDataType) -> np.ndarray:
        """Returns the precision in each of the data points of the specified type.

        Parameters
        ----------
        data_type: NISTDataType

        Returns
        -------
        numpy.ndarray:
            The precision in each data point.
        """
        return self._precisions[data_type]

from __future__ import annotations

import copy
import logging
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import sympy as sp
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field, model_validator
from pyenzyme import DataTypes, EnzymeMLDocument
from rich.console import Console
from rich.table import Table

from calipytion.model import (
    CalibrationModel,
    CalibrationRange,
    Standard,
    UnitDefinition,
)
from calipytion.tools.fitter import Fitter
from calipytion.tools.utility import pubchem_request_molecule_name
from calipytion.units import C

LOGGER = logging.getLogger(__name__)

# Ignore Pydantic serializer warnings for now due to required workaround
# for AnIML <F>, <I>, <S>, and <Boolean> tags raising a warning about
# type mismatches like `FloatType` vs `float`.
warnings.filterwarnings(
    "ignore", category=UserWarning, message="^Pydantic serializer warnings:"
)


class Calibrator(BaseModel):
    """
    Class to handle the calibration process and including creation, fitting, comparison,
    visualization, and selection of calibration models.
    """

    molecule_id: str = Field(
        description="Unique identifier of the given object.",
        pattern=r"^[a-zA-Z][a-zA-Z0-9^\+\-\*/=<>^\|&%!~]*$",
    )

    pubchem_cid: int = Field(
        description="PubChem Compound Identifier",
    )

    molecule_name: str = Field(
        description="Name of the molecule",
    )

    concentrations: list[float] = Field(
        description="Concentrations of the standard",
    )

    wavelength: float | None = Field(
        description="Wavelength of the measurement",
        default=None,
    )

    conc_unit: UnitDefinition = Field(
        description="Concentration unit",
    )

    signals: list[float] = Field(
        description="Measured signals, corresponding to the concentrations",
    )

    models: list[CalibrationModel] = Field(
        description="Models used for fitting", default=[], validate_default=True
    )

    cutoff: float | None = Field(
        default=None,
        description=(
            "Upper cutoff value for the measured signal. All signals above this value"
            " will be ignored during calibration"
        ),
    )

    standard: Standard | None = Field(
        default=None,
        description="Result oriented object, representing the data and the chosen model.",
    )

    @model_validator(mode="before")
    @classmethod
    def get_molecule_name(cls, data: Any) -> Any:
        """Retrieves the molecule name from the PubChem database based on the PubChem CID."""

        if "molecule_name" not in data:
            data["molecule_name"] = pubchem_request_molecule_name(data["pubchem_cid"])
        return data

    @model_validator(mode="after")
    def initialize_models(self):
        """
        Loads the default models if no models are provided and initializes the models
        with the according 'molecule_id'.
        """
        if not self.models:
            from calipytion.tools.equations import (
                cubic_model,
                linear_model,
                quadratic_model,
            )

            modified_models = []
            for m in [linear_model, quadratic_model, cubic_model]:
                model = copy.deepcopy(m)
                model.signal_law = model.signal_law.replace(
                    "concentration", self.molecule_id
                )
                model.molecule_id = self.molecule_id
                modified_models.append(model)

            self.models = modified_models

            return self

        return self

    def model_post_init(self, __context: Any) -> None:
        self._apply_cutoff()

    def add_model(
        self,
        name: str,
        signal_law: str,
        init_value: float = 1,
        lower_bound: float = -1e-6,
        upper_bound: float = 1e6,
    ) -> CalibrationModel:
        """Add a model to the list of models used for calibration."""

        assert (
            self.molecule_id in signal_law
        ), f"Equation must contain the symbol of the molecule to be calibrated ('{self.molecule_id}')"

        model = CalibrationModel(
            molecule_id=self.molecule_id,
            name=name,
            signal_law=signal_law,
        )

        for symbol in self._get_free_symbols(signal_law):
            if symbol == self.molecule_id:
                model.molecule_id = symbol
                continue

            model.add_to_parameters(
                symbol=symbol,
                init_value=init_value,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

        self.models.append(model)

        return model

    def get_model(self, model_name: str) -> CalibrationModel:
        """Returns a model by its name."""

        for model in self.models:
            if model.name == model_name:
                return model

        raise ValueError(f"Model '{model_name}' not found")

    def calculate_concentrations(
        self,
        model: CalibrationModel | str,
        signals: list[float],
        extrapolate: bool = False,
    ) -> list[float]:
        """Calculates the concentration from a given signal using a calibration model.

        When calculating the concentration, the model is used to calculate the roots of the
        equation for the given signals. The model is parameterized by previous fitting using the
        `fit_models` method.
        By default extrapolation is disabled, meaning that the concentration is only calculated
        within the calibration range of the model. If extrapolation is enabled, the concentration
        can be calculated outside the calibration range. Be aware that the results might be
        unreliable.

        Args:
            model (CalibrationModel | str): The model object or name which should be used.
            signals (list[float]): The signals for which the concentration should be calculated.
            extrapolate (bool, optional): Whether to extrapolate the concentration outside the
                calibration range. Defaults to False.

        Returns:
            list[float]: The calculated concentrations.
        """
        if not isinstance(model, CalibrationModel):
            model = self.get_model(model)

        np_signals = np.array(signals)

        assert model.calibration_range, "Calibration range not set."

        lower_bond = model.calibration_range.conc_lower
        upper_bond = model.calibration_range.conc_upper

        cal_model = Fitter.from_calibration_model(model)

        concs, bracket = cal_model.calculate_roots(
            y=np_signals,
            lower_bond=lower_bond,
            upper_bond=upper_bond,
            extrapolate=extrapolate,
        )

        # give warning if any concentration is nan
        if np.isnan(concs).any() and not extrapolate:
            LOGGER.warning(
                "⚠️ Some concentrations could not be calculated and were replaced with nan "
                f"values, since the provided signal is outside the calibration range {bracket}. "
                "To calculate the concentration outside the calibration range, set "
                "'extrapolate=True'."
            )

        # Update or create standard object based on used model
        if self.standard:
            self._update_model_of_standard(model)

        return concs.tolist()

    def apply_to_enzymeml(
        self,
        enzmldoc: EnzymeMLDocument,
        extrapolate: bool = False,
        silent: bool = False,
    ):
        """Applies the calibrator to an EnzymeML document if the species_id and molecule_id
        match between the EnzymeML Document and the calibrator.

        Args:
            enzmldoc (EnzymeMLDocument): The EnzymeML document to apply the calibrator to.
            extrapolate (bool, optional): Whether to extrapolate the concentration outside the
                calibration range. Defaults to False.
            silent (bool, optional): Silences the print output. Defaults to False.

        Raises:
            AssertionError: If no standard with a fitted calibration model is found.
            AssertionError: If the data is already in concentration values.
            AssertionError: If the units of the measured data and the calibration model do not match.
        """

        assert self.standard, "No standard object found."
        assert self.standard.result, "No model found in the standard object."

        converted_count = 0
        for measurement in enzmldoc.measurements:
            for measured_species in measurement.species_data:
                if measured_species.species_id == self.molecule_id:
                    assert measured_species.data_type != DataTypes.CONCENTRATION, """
                        The data seems to be already in concentration values.
                    """
                    # assert units are the same
                    assert measured_species.data_unit.name == self.conc_unit.name, f"""
                    The unit of the measured data ({measured_species.data_unit.name}) is not 
                    the same as the unit of the calibration model ({self.conc_unit.name}).
                    """

                    signals = measured_species.data
                    measured_species.data = self.calculate_concentrations(
                        self.standard.result, signals, extrapolate
                    )
                    measured_species.data_type = DataTypes.CONCENTRATION

                    converted_count += 1

        symbol = "✅" if converted_count > 0 else "❌"
        if not silent:
            print(f"{symbol} Applied calibration to {converted_count} measurements")

    def export_to_animl(self, silent: bool = False) -> Any:
        """Exports measurements and models to an AnIML document.

        Args:
            wavelength_nm (float): The wavelength of the measurement in nm.

        Raises:
            AssertionError: If no standard object is found.
            AssertionError: If no model is found in the standard object.

        Returns:
            animl_document: The AnIML document.
        """
        from calipytion.ioutils.animlio import get_animl_document, map_standard_to_animl

        assert self.standard, "No standard object found."
        assert self.standard.result, "No model found in the standard object."

        self.standard.wavelength = self.wavelength

        animl_document = get_animl_document()
        animl_document = map_standard_to_animl(self.standard, animl_document)

        if not silent:
            print("✅ Applied data to AnIML document")

        return animl_document

    def _update_model_of_standard(self, model: CalibrationModel) -> None:
        """Updates the model of the standard object with the given model."""

        assert self.standard, "No standard object found."
        assert (
            self.molecule_id == self.standard.molecule_id
        ), "The molecule id of the Calibrator and the Standard object must be the same."

        self.standard.result = model

    @classmethod
    def from_csv(
        cls,
        path: str,
        molecule_id: str,
        conc_unit: UnitDefinition,
        pubchem_cid: int,
        concentration_column_name: str,
        molecule_name: str | None = None,
        cutoff: Optional[float] = None,
        wavelength: Optional[float] = None,
    ):
        """Reads the data from a CSV file and initializes the Calibrator object.

        Args:
            path (str): Path to the CSV file.
            molecule_id (str): Unique identifier of the molecule.
            conc_unit (UnitDefinition): Concentration unit.
            pubchem_cid (int): PubChem Compound Identifier.
            concentration_column_name (str): Name of the column containing the concentrations.
            molecule_name (str, optional): Name of the molecule. Defaults to None.
            cutoff (float, optional): Cutoff value for the signals If a signal is above this value, it
                will be not considered for the calibration. Defaults to None.
            wavelength (float, optional): Wavelength of the measurement. Defaults to None.

        Returns:
            Calibrator: The Calibrator object.
        """

        df = pd.read_csv(path)

        conc_values = df[concentration_column_name].values
        all_other_columns = df.columns.drop(concentration_column_name)
        signals_matrix = df[all_other_columns].values
        signals = signals_matrix.flatten()

        concentrations = np.repeat(conc_values, signals_matrix.shape[1])

        return cls(
            molecule_id=molecule_id,
            conc_unit=conc_unit,
            pubchem_cid=pubchem_cid,
            concentrations=concentrations.tolist(),
            signals=signals.tolist(),
            molecule_name=molecule_name,
            cutoff=cutoff,
            wavelength=wavelength,
        )

    @classmethod
    def from_excel(
        cls,
        path: str,
        molecule_id: str,
        conc_unit: UnitDefinition,
        pubchem_cid: int,
        molecule_name: str | None = None,
        cutoff: Optional[float] = None,
        wavelength: Optional[float] = None,
        sheet_name: Optional[str | int] = 0,
        skip_rows: Optional[int] = 0,
    ):
        """Reads the data from an Excel file and initializes the Calibrator object.
        The leftmost column is expected to contain the concentrations. All other columns
        are expected to contain the signals of the respective samples.

        Args:
            path (str): Path to the Excel file.
            molecule_id (str): Unique identifier of the molecule.
            molecule_name (str): Name of the molecule.
            conc_unit (UnitDefinition): Concentration unit.
            pubchem_cid (int): PubChem Compound Identifier.
            cutoff (float, optional): Cutoff value for the signals. Defaults to None.
            wavelength (float, optional): Wavelength of the measurement. Defaults to None.
            sheet_name (str | int, optional): Name of the sheet in the Excel file. Defaults to 0.
            skip_rows (int, optional): Number of rows to skip at the beginning of the sheet. Defaults to 0.

        Returns:
            Calibrator: The Calibrator object.
        """

        df = pd.read_excel(path, sheet_name=sheet_name, header=None, skiprows=skip_rows)

        signals = df.iloc[:, 1:].values  # type: ignore
        n_reps = signals.shape[1]
        signals = signals.flatten().tolist()

        concs = df.iloc[:, 0].values  # type: ignore
        concs = np.repeat(concs, n_reps)  # type: ignore
        concs = concs.flatten().tolist()

        args = {
            "molecule_id": molecule_id,
            "pubchem_cid": pubchem_cid,
            "concentrations": concs,
            "signals": signals,
            "conc_unit": conc_unit,
            "cutoff": cutoff,
            "wavelength": wavelength,
        }

        # Add molecule_name only if it's not None
        if molecule_name is not None:
            args["molecule_name"] = molecule_name

        return cls(**args)

    def add_model_to_standard(self, model: CalibrationModel | str) -> None:
        """Adds a model to the result object of the standard.

        Args:
            model (CalibrationModel | str): The model object or name which should be added to the standard.

        Raises:
            ValueError: If the model has not been fitted yet.
            ValueError: If no standard object is found.
        """

        if isinstance(model, str):
            model = self.get_model(model)

        if not model.was_fitted:
            raise ValueError("Model has not been fitted yet. Run 'fit_models' first.")

        if self.standard is None:
            raise ValueError(
                "No standard object found. Create a standard first by calling 'create_standard'."
            )

        self.standard.result = model

    @classmethod
    def from_json(
        cls,
        path: str,
        cutoff: Optional[float] = None,
    ) -> Calibrator:
        """Reads the data from a JSON Standard file and initializes the Calibrator object.

        Args:
            path (str): Path to the JSON file.
            cutoff (Optional[float], optional): Cutoff value for the signals.
                Higher signals will be ignored. Defaults to None.

        Returns:
            Calibrator: The Calibrator object.
        """
        import json

        with open(path, "r") as file:
            standard = Standard(**json.load(file))

        return cls.from_standard(standard, cutoff)

    @classmethod
    def from_standard(
        cls,
        standard: Standard,
        cutoff: float | None = None,
    ) -> Calibrator:
        """Initialize the Calibrator object from a Standard object.

        Args:
            standard (Standard): The Standard object to be used for calibration.
            cutoff (float | None, optional): Whether to apply a cutoff value to the signals.
                This filters out all signals and corresponding concentration above the
                cutoff value. Defaults to None.

        Raises:
            ValueError: If the number of concentrations and signals are not the same.
            ValueError: If all samples do not have the same concentration unit.

        Returns:
            Calibrator: The Calibrator object.
        """

        # get concentrations and corresponding signals as lists
        concs = [sample.concentration for sample in standard.samples]
        signals = [sample.signal for sample in standard.samples]
        if not len(concs) == len(signals):
            raise ValueError("Number of concentrations and signals must be the same")

        # verify that all samples have the same concentration unit
        if not all(
            [
                sample.conc_unit == standard.samples[0].conc_unit
                for sample in standard.samples
            ]
        ):
            raise ValueError("All samples must have the same concentration unit")
        conc_unit = standard.samples[0].conc_unit

        if standard.result:
            models = [standard.result]
        else:
            models = []

        return cls(
            molecule_id=standard.molecule_id,
            pubchem_cid=standard.pubchem_cid,
            molecule_name=standard.molecule_name,
            concentrations=concs,
            signals=signals,
            conc_unit=conc_unit,
            models=models,
            cutoff=cutoff,
            standard=standard,
        )

    def fit_models(self, silent: bool = False):
        """Fits all models to the given data.

        Args:
            silent (bool, optional): Silences the print output of
                the fitter. Defaults to False.
        """

        for model in self.models:
            # Set the calibration range of the model
            model.calibration_range = CalibrationRange(
                conc_lower=min(self.concentrations),
                conc_upper=max(self.concentrations),
                signal_lower=min(self.signals),
                signal_upper=max(self.signals),
            )

            y_data = np.array(self.signals)
            x_data = np.array(self.concentrations)

            # Fit model
            fitter = Fitter.from_calibration_model(model)
            statisctics = fitter.fit(
                y=y_data, x=x_data, indep_var_symbol=self.molecule_id
            )

            # Set the fit statistics
            model.statistics = statisctics
            model.was_fitted = True

        # Sort models by AIC
        self.models = sorted(self.models, key=lambda x: x.statistics.aic)

        if not silent:
            print("✅ Models have been successfully fitted.")
            self.print_result_table()

    def print_result_table(self) -> None:
        """
        Prints a table with the results of the fitted models.
        """

        console = Console()

        table = Table(title="Model Overview")
        table.add_column("Model Name", style="magenta")
        table.add_column("AIC", style="cyan")
        table.add_column("R squared", style="cyan")
        table.add_column("RMSD", style="cyan")
        table.add_column("Equation", style="cyan")
        table.add_column("Relative Parameter Standard Errors", style="cyan")

        for model in self.models:
            param_string = ""
            for param in model.parameters:
                if not param.stderr:
                    stderr = "n.a."
                else:
                    stderr = abs(param.stderr / param.value)
                    stderr = str(round(stderr * 100, 1)) + "%"
                param_string += f"{param.symbol}: {stderr}, "

            table.add_row(
                model.name,
                str(round(model.statistics.aic)),
                str(round(model.statistics.r2, 4)),
                str(round(model.statistics.rmsd, 4)),
                model.signal_law,
                param_string,
            )

        console.print(table)

    def visualize(self) -> None:
        assert any(
            [model.was_fitted for model in self.models]
        ), "No model has been fitted yet. Run 'fit_models' first."

        """
        Visualizes the calibration curve and the residuals of the models.
        """
        fig = make_subplots(
            rows=1,
            cols=2,
            x_title=f"{self.molecule_name} / {self.conc_unit.name}",
            subplot_titles=[
                "Standard",
                "Model Residuals",
            ],
            horizontal_spacing=0.15,
        )
        colors = px.colors.qualitative.Plotly

        buttons = []
        if self.standard:
            fig = self._traces_from_standard(fig)
        else:
            fig.add_trace(
                go.Scatter(
                    x=self.concentrations,
                    y=self.signals,
                    name=f"{self.molecule_name}",
                    mode="markers",
                    marker=dict(color="#000000"),
                    visible=True,
                    customdata=[f"{self.molecule_name} standard"],
                ),
                col=1,
                row=1,
            )

        for model, color in zip(self.models, colors):
            fitter = Fitter.from_calibration_model(model)

            assert model.calibration_range, "Calibration range not set."

            smooth_x = np.linspace(
                model.calibration_range.conc_lower,
                model.calibration_range.conc_upper,
                100,
            ).tolist()

            params = {param.symbol: param.value for param in model.parameters}
            params[model.molecule_id] = np.array(smooth_x)  # type: ignore

            model_pred = fitter.lmfit_model.eval(**params)

            fitter.fit(self.signals, self.concentrations, model.molecule_id)
            residuals = fitter.lmfit_result.residual

            # Add model traces
            fig.add_trace(
                go.Scatter(
                    x=smooth_x,
                    y=model_pred,
                    name=f"{model.name} model",
                    mode="lines",
                    marker=dict(color=color),
                    visible=False,
                    customdata=[f"{model.name} model"],
                ),
                col=1,
                row=1,
            )

            # Add residual traces
            fig.add_trace(
                go.Scatter(
                    x=self.concentrations,
                    y=residuals,
                    name="Residuals",
                    mode="markers",
                    marker=dict(color=color),
                    hoverinfo="skip",
                    visible=False,
                    customdata=[f"{model.name} model"],
                ),
                col=2,
                row=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=self.concentrations,
                    y=np.zeros(len(self.concentrations)),
                    line=dict(color="grey", width=2, dash="dash"),
                    visible=True,
                    customdata=[f"{model.name} model"],
                    showlegend=False,
                ),
                col=2,
                row=1,
            )

        buttons.append(
            dict(
                method="update",
                args=[
                    dict(
                        visible=self._visibility_mask(
                            visible_traces=[f"{self.molecule_name} standard"],
                            fig_data=fig.data,
                        )
                    )
                ],
                label=f"{self.molecule_name} standard",
            ),
        )

        for model in self.models:
            buttons.append(
                dict(
                    method="update",
                    args=[
                        dict(
                            visible=self._visibility_mask(
                                visible_traces=[
                                    f"{model.name} model",
                                    f"{self.molecule_name} standard",
                                ],
                                fig_data=fig.data,
                            ),
                            title=f"{model.name} model",
                        )
                    ],
                    label=f"{model.name} model",
                )
            )

        all_traces = [f"{model.name} model" for model in self.models]
        all_traces.append(f"{self.molecule_name} standard")
        buttons.append(
            dict(
                method="update",
                label="all",
                args=[
                    dict(
                        visible=self._visibility_mask(
                            visible_traces=all_traces,
                            fig_data=fig.data,
                        )
                    )
                ],
            )
        )

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0,
                    y=1.2,
                    xanchor="left",
                    yanchor="top",
                    buttons=buttons,
                )
            ],
            margin=dict(l=20, r=20, t=100, b=60),
            template="simple_white",
        )

        if self.wavelength:
            signal_label = f"(E<sub>{self.wavelength:.0f} nm</sub>)"
        else:
            signal_label = "(a.u.)"

        fig.update_yaxes(
            title_text=f"{self.molecule_name} {signal_label}", row=1, col=1
        )
        fig.update_yaxes(
            title_text=f"Residuals {self.molecule_name} {signal_label}", row=1, col=2
        )
        fig.update_traces(hovertemplate="Signal: %{y:.2f}")

        config = {
            "toImageButtonOptions": {
                "format": "png",  # one of png, svg, jpeg, webp
                "filename": f"{self.molecule_name}_calibration_curve",
                # "height": 600,
                # "width": 700,
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

        # save figure as plotly json
        # from plotly.io import write_json
        # write_json(fig, "usage_plot.json")

        return fig.show(config=config)

    def _traces_from_standard(self, fig: go.Figure):
        for sample in self.standard.samples:
            if not hasattr(sample, "id"):
                sample_id = f"Sample {self.standard.samples.index(sample) + 1}"
            else:
                sample_id = sample.id
            fig.add_trace(
                go.Scatter(
                    x=[sample.concentration],
                    y=[sample.signal],
                    name=sample_id,
                    mode="markers",
                    marker=dict(color="#000000"),
                    visible=True,
                    customdata=[f"{self.standard.molecule_name} standard"],
                    showlegend=False,
                ),
                col=1,
                row=1,
            )

        return fig

    def create_standard(
        self,
        model: CalibrationModel,
        ph: float,
        temperature: float,
        temp_unit: UnitDefinition = C,
        retention_time: Optional[float] = None,
    ) -> Standard:
        """Creates a standard object with the given model, pH, and temperature.

        Args:
            model (CalibrationModel): The fitted model to be used for the standard.
            ph (float): The pH value of the standard.
            temperature (float): The temperature of the standard.
            temp_unit (str): The unit of the temperature. Defaults to "C".
            retention_time (float, optional): Retention time of the molecule. Defaults to None.

        Raises:
            ValueError: If the model has not been fitted yet.

        Returns:
            Standard: The created standard object.
        """

        if not model.was_fitted:
            raise ValueError("Model has not been fitted yet. Run 'fit_models' first.")

        standard = Standard(
            molecule_id=self.molecule_id,
            pubchem_cid=self.pubchem_cid,
            molecule_name=self.molecule_name,
            wavelength=self.wavelength,
            ph=ph,
            temp_unit=temp_unit,
            temperature=temperature,
            samples=[],
            result=model,
            retention_time=retention_time,
            ld_id=f"https://pubchem.ncbi.nlm.nih.gov/compound/{self.pubchem_cid}",
        )

        for conc, signal in zip(self.concentrations, self.signals):
            standard.add_to_samples(
                concentration=conc, signal=signal, conc_unit=self.conc_unit
            )

        self.standard = standard

        return standard

    @staticmethod
    def _visibility_mask(visible_traces: list, fig_data: list) -> list:
        return [
            any(fig["customdata"][0] == trace for trace in visible_traces)
            for fig in fig_data
        ]

    def _get_free_symbols(self, equation: str) -> list[str]:
        """Gets the free symbols from a sympy equation and converts them to strings."""

        sp_eq = sp.sympify(equation)
        symbols = list(sp_eq.free_symbols)

        return [str(symbol) for symbol in symbols]

    def _apply_cutoff(self):
        """Applies the cutoff value to the signals and concentrations."""

        if self.cutoff:
            below_cutoff_idx = [
                idx for idx, signal in enumerate(self.signals) if signal < self.cutoff
            ]

            self.concentrations = [self.concentrations[idx] for idx in below_cutoff_idx]
            self.signals = [self.signals[idx] for idx in below_cutoff_idx]

    def visualize_static(self, show: bool = True) -> None:
        """Creates a static visualization of the calibration curve and residuals using matplotlib.

        Args:
            show (bool): Whether to display the plot. Set to False when using in notebooks.
                        Defaults to True.
        """
        assert any(
            [model.was_fitted for model in self.models]
        ), "No model has been fitted yet. Run 'fit_models' first."

        import matplotlib.pyplot as plt

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot measured data points
        ax1.scatter(
            self.concentrations, self.signals, color="black", label="Measurements"
        )

        # Plot fitted models
        for model in self.models:
            fitter = Fitter.from_calibration_model(model)
            assert model.calibration_range, "Calibration range not set."

            # Generate smooth x data for model line
            smooth_x = np.linspace(
                model.calibration_range.conc_lower,
                model.calibration_range.conc_upper,
                100,
            )

            # Calculate model predictions
            params = {param.symbol: param.value for param in model.parameters}
            params[model.molecule_id] = smooth_x
            model_pred = fitter.lmfit_model.eval(**params)

            # Plot model line
            ax1.plot(
                smooth_x,
                model_pred,
                label=f"{model.name} (R² = {model.statistics.r2:.3f})",
            )

            # Calculate and plot residuals
            fitter.fit(self.signals, self.concentrations, model.molecule_id)
            residuals = fitter.lmfit_result.residual
            ax2.scatter(self.concentrations, residuals, label=model.name)

        # Add zero line to residuals plot
        ax2.axhline(y=0, color="grey", linestyle="--", alpha=0.5)

        # Set titles and labels
        if self.wavelength:
            signal_label = f"Signal (E{self.wavelength:.0f} nm)"
        else:
            signal_label = "Signal (a.u.)"

        ax1.set_xlabel(f"{self.molecule_name} ({self.conc_unit.name})")
        ax1.set_ylabel(signal_label)
        ax1.set_title("Calibration Curve")
        ax1.legend()

        ax2.set_xlabel(f"{self.molecule_name} ({self.conc_unit.name})")
        ax2.set_ylabel(f"Residuals {signal_label}")
        ax2.set_title("Residuals")
        ax2.legend()

        # Adjust layout
        plt.tight_layout()

        if show:
            plt.show()

        return fig


if __name__ == "__main__":
    from calipytion.units import uM

    # initialize the calibrator
    calibrator = Calibrator.from_csv(
        path="/Users/max/Documents/GitHub/Range-2024/data/abts-calibration.csv",
        molecule_id="s0",
        molecule_name="ABTS",
        pubchem_cid=5360881,
        cutoff=3.2,
        wavelength=340,
        conc_unit=uM,
        concentration_column_name="ABTS (µmol/l)",
    )

    # define the models to be fitted as potential signal laws
    calibrator.models = []

    linear = calibrator.add_model(
        name="linear",
        signal_law="a * s0 + b",
    )
    quadratic = calibrator.add_model(
        name="quadratic",
        signal_law="a * s0**2 + b * s0 + c",
    )
    cubic = calibrator.add_model(
        name="cubic",
        signal_law="a * s0**3 + b * s0**2 + c * s0 + d",
    )

    # fit the models
    calibrator.fit_models()
    # calibrator.visualize()
    # calibrator.visualize_static()

    calibrator.create_standard(
        linear,
        ph=7.4,
        temperature=25,
        temp_unit=C,
    )
    animl = calibrator.export_to_animl()
    print(animl)

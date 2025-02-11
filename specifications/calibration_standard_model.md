---
repo: "https://github.com/FAIRChemistry/CaliPytion"
prefix: "calipy"
---

# Calibration Data Model

This data model structures data of standard measurements of a molecule.

## Root objects

### Standard

The `Standard` contains information on the molecule for which the calibration was performed, its standard measurements, measurement conditions, as well as the fitted calibration model.

- **molecule_id**
  - Type: string
  - Description: Short ID how the molecule should be referenced in equations. E.g. `s1`.
- **pubchem_cid**
  - Type: integer
  - Description: PubChem Compound Identifier.
- **molecule_name**
  - Type: string
  - Description: Name of the molecule.
- **ph**
  - Type: float
  - Description: pH value of the solution.
- **temperature**
  - Type: float
  - Description: Temperature during calibration.
- **temp_unit**
  - Type: UnitDefinition
  - Description: Temperature unit.
- retention_time
  - Type: float
  - Description: Retention time of the molecule in minutes.
- wavelength
  - Type: float
  - Description: Detection wavelength in nm.
- samples
  - Type: Sample[]
  - Description: Measured signal, at a given concentration of the molecule.
- result
  - Type: CalibrationModel
  - Description: The model that was used for concentration determination.

### Sample

A `Sample` describes individual measured signal-concentration pairs of a molecule.

- **concentration**
  - Type: float
  - Description: Concentration of the molecule.
- **conc_unit**
  - Type: UnitDefinition
  - Description: Concentration unit.
- **signal**
  - Type: float
  - Description: Measured signals at a given concentration of the molecule.

### CalibrationModel

The `CalibrationModel` describes the calibration model which was fitted to the calibration data. The calibration model consists of the signal law and parameters of the equation. The calibration range defines the concentration and signal bounds in which the calibration model is valid.

- **name**
  - Type: string
  - Description: Name of the calibration model.
- molecule_id
  - Type: string
  - Description: ID of the molecule like ChEBI ID.
- signal_law
  - Type: string
  - Description: Law describing the signal intensity as a function of the concentration.
- parameters
  - Type: Parameter[]
  - Description: Parameters of the calibration equation.
- was_fitted
  - Type: boolean
  - Description: Indicates if the model was fitted to the data.
  - default: False
- calibration_range
  - Type: CalibrationRange
  - Description: Concentration and signal bounds in which the calibration model is valid.
- statistics
  - Type: FitStatistics
  - Description: Fit statistics of the calibration model.

### CalibrationRange

THe `CalibrationRange` defines the concentration and signal bounds in which the calibration model is valid.

- **conc_lower**
  - Type: float
  - Description: Lower concentration bound of the model.
- **conc_upper**
  - Type: float
  - Description: Upper concentration bound of the model.
- **signal_lower**
  - Type: float
  - Description: Lower signal bound of the model.
- **signal_upper**
  - Type: float
  - Description: Upper signal bound of the model.

### FitStatistics

The `FitStatistics` contains statistical parameters of the fitted calibration model.

- aic
  - Type: float
  - Description: Akaike information criterion.
- bic
  - Type: float
  - Description: Bayesian information criterion.
- r2
  - Type: float
  - Description: Coefficient of determination.
- rmsd
  - Type: float
  - Description: Root mean square deviation.

### Parameter

A `Parameter` describes the value, standard error, and bounds of a parameter which is part of the signal law.

- symbol
  - Type: string
  - Description: Name of the parameter.
- value
  - Type: float
  - Description: Value of the parameter.
- init_value
  - Type: float
  - Description: Initial value of the parameter.
- stderr
  - Type: float
  - Description: 1-sigma standard error of the parameter.
- lower_bound
  - Type: float
  - Description: Lower bound of the parameter prior to fitting.
- upper_bound
  - Type: float
  - Description: Upper bound of the parameter prior to fitting.

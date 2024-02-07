# CaliPytion - Tool for Calibration Model Creation, Application, and Documentation

[![Generate API](https://github.com/FAIRChemistry/CaliPytion/actions/workflows/generate_api.yaml/badge.svg)](https://github.com/FAIRChemistry/CaliPytion/actions/workflows/generate_api.yaml)
[![Documentation](https://github.com/FAIRChemistry/CaliPytion/actions/workflows/make_docs.yaml/badge.svg)](https://github.com/FAIRChemistry/CaliPytion/actions/workflows/make_docs.yaml)
[![Tests](https://github.com/FAIRChemistry/CaliPytion/actions/workflows/tests.yaml/badge.svg)](https://github.com/FAIRChemistry/CaliPytion/actions/workflows/tests.yaml)



## 🛤 What is CaliPytion?
CaliPytion is a Python tool designed for analyzing the relationship between measured signals and concentrations using various calibration models. It operates on the `Standard` data model, contining data of calibration measurements and their respective conditions, and primarily employs the `Calibrator` object. This object is adept at fitting different polynomial models to calibration data using least-squares minimization.

Key functionalities of CaliPytion include:

- __Model Fitting and Visualization__: Automatically fits different polynomial models to the data and provides interactive plots for visual comparison of these models.
- __Detailed Reporting__: Generates reports including the Akaike Information Criterion (AIC) and root-mean-square deviation (RMSD), aiding in the comparison of models based on fit quality.
- __Model Selection and Application__: Integrates the chosen model into the Standard object, using it for precise concentration calculations from measured signals. Signals outside the valid calibration are replaced with with NaN values, if model extrapolation is disabled.
- __Export and Documentation__: Allows for the export of calibration models in multiple formats like Analytical Information Markup Language (AnIML), YAML, JSON, or XML, facilitating documentation and sharing.

## ⚡️ Quick Start

Get started with CaliPytion by cloning this repository:

```Bash
git clone https://github.com/FAIRChemistry/CaliPytion/

```

Or install from PyPi:

```Bash
pip install CaliPytion
```

## 🔖 Example Code

coming soon

## ⚖️ License

Copyright (c) 2023 FAIR Chemistry

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

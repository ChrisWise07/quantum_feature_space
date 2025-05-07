# Quantum Feature Spaces




## Overview
This is the repository for the paper *'Quantum Feature Space of a Qubit Coupled to an Arbitrary Bath'*, found here: https://arxiv.org/abs/2505.03397 

This project  introduces and implements the **Quantum Feature Space**, an efficient parameterization for describing the impact of environmental noise on qubit dynamics. It serves as an alternative to traditional power spectral density methods or computationally intensive neural network approaches for inferring noise operators.

The core idea is that this feature space provides a compact yet informative description of the noise's effect. This repository demonstrates how:

* Distances in this space can classify different noise types.
* The feature space can be used as input for machine learning classifiers (e.g., Random Forest) to determine noise characteristics.
* The mapping between control pulses and the feature space can be analyzed.

Code is provided for generating feature spaces, performing classification tasks, and related analyses.

## Installation

1. **Clone the repository:**

    ```bash
    git clone <YOUR_REPOSITORY_URL> # Replace with your repo URL
    cd quantum-feature-spaces       # Or your actual repository directory name
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**
    Ensure you have Python 3.x installed. Install the required packages using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Generation

To run specific data generation scripts, execute them as modules from the project's root directory. For example, to generate the real CPMG (Carr-Purcell-Meiboom-Gill) dataset:

```bash
python -m data_gen.gen_real_CPMG_data
```

There are Juypter notebooks which can be run directly in a notebook environment.

All other files are standard Python scripts. You can run them from the command line via

```bash
python <script_name>.py
```

### Custom Noise Models
This project allows for the integration and testing of custom noise models within the quantum simulations. If you are interested in exploring, modifying, or adding new noise models, please see the following core files:
- qubit_sim/noise_gen_funcs.py: Contains utility functions related to noise generation processes. This is where you define noise process values as a function of time (in this case, a torch tensor).
- qubit_sim/noise_gen_classes.py: Defines classes representing different noise models that can be applied during simulations. Essentially, the classes here are wrappers around the functions in noise_gen_funcs.py. You can create new classes that inherit from the base class and implement your own noise generation logic.

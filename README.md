# DeepFMKit: A Toolkit for Deep Frequency Modulation Interferometry

A high-performance Python framework for simulating, processing, and analyzing data from Deep Frequency Modulation Interferometry (DFMI) experiments.

## About The Project

Deep Frequency Modulation Interferometry (DFMI) is a laser-based metrology technique used for high-precision displacement sensing and absolute distance measurement. This toolkit, **DeepFMKit**, provides a complete software environment for researchers and engineers working with DFMI.

It is designed to handle the entire experimental workflow, from simulating complex interferometric signals with realistic noise sources to performing robust, high-speed non-linear least squares (NLS) fits to extract physical parameters.

### Key Features

*   **Realistic Simulation:** Create synthetic DFMI data using a detailed physical model that includes laser frequency noise, amplitude noise, and dynamic arm length modulation.
*   **Data Handling:** Load and save experimental data from custom text file formats.
*   **High-Performance Fitting:** A highly optimized NLS fitting engine to extract key observables (`m`, `phi`, `psi`, `amp`) from raw data.
*   **Parallel Processing:** Built-in support for parallelizing fits across multiple CPU cores to dramatically accelerate the processing of large datasets.
*   **Analysis & Visualization:** Integrated tools for post-processing results, including spectral analysis, and a suite of plotting functions.
*   **Modular Architecture:** A clean, object-oriented design that separates concerns, making the toolkit easy to understand and extend.

## Getting Started

### Prerequisites

*   Python 3.9+
*   Standard scientific libraries (NumPy, SciPy, Pandas, Matplotlib)

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/your_username/DeepFMKit.git
    cd DeepFMKit
    ```
2.  Install the required packages. It's recommended to do this in a virtual environment.
    ```sh
    pip install -r requirements.txt
    ```

## Quick Start: A Complete Workflow

The following example demonstrates the primary workflow: creating a simulation, generating data, running a fit, and plotting the results.

```python
from DeepFMKit.core import DeepFitFramework
import matplotlib.pyplot as plt

dff = DeepFitFramework()

# Create a `DeepSimObject` describing an interferometer
label = "dynamic"
dff.new_sim(label)
dff.sims[label].m = 6.0
dff.sims[label].f_mod = 1000
dff.sims[label].f_samp = int(200e3)
dff.sims[label].f_n = 1e6
dff.sims[label].arml_mod_f = 1.0
dff.sims[label].arml_mod_amp = 1e-9
dff.sims[label].arml_mod_n = 1e-12
dff.sims[label].fit_n = 10

# Simulate DFM interferometer with test mass motion and a reference channel
dff.simulate(label, n_seconds=10, simulate="dynamic", ref_channel=True)

# Print system information
dff.sims[label].info()

# Perform non-linear least squares fitting on data
for i, key in enumerate(dff.sims):
    dff.fit(label=key, fit_label=f'ch{i}')

# Plot results
ax = dff.plot()
plt.show()
```
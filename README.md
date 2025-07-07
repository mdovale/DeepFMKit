# DeepFMKit: A Toolkit for Deep Frequency Modulation Interferometry

A high-performance Python framework for simulating, processing, and analyzing data from Deep Frequency Modulation Interferometry (DFMI) experiments.

## About The Project

Deep Frequency Modulation Interferometry (DFMI) is a laser-based metrology technique used for high-precision displacement sensing and absolute distance measurement. This toolkit, **DeepFMKit**, provides a complete software environment for researchers and engineers working with DFMI.

It is designed to handle the entire experimental workflow, from simulating complex interferometric signals with realistic noise sources to performing robust high-speed DFMI-readout. For this purpose, it features two mathematical engines: non-linear least squares (NLS) fits acting on the frequency-domain, and an Extended Kalman Filter acting on the time-domain.

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
import DeepFMKit.core as dfm
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc

# Instantiate the main framework
dff = dfm.DeepFitFramework()

# --- 1. Define the Single, Shared Laser Source ---
laser_config = dfm.LaserConfig(label="main_laser")
laser_config.f_mod = 1000 # Modulation frequency (Hz)

# --- 2. Define the Main Interferometer ---
main_ifo_config = dfm.InterferometerConfig(label="dynamic_ifo")
main_ifo_config.ref_arml = 0.1 # Reference arm length (m)
main_ifo_config.meas_arml = 0.3 # Measurement arm length (m)

# --- 3. Set Modulation Depth by Adjusting Laser's `df` ---
m_target = 6.0 # Target effective modulation index (rad)
opd = main_ifo_config.meas_arml - main_ifo_config.ref_arml # Optical pathlength difference (m)
df_required = (m_target * sc.c) / (2 * np.pi * opd) # Required laser modulation amplitude (Hz)
laser_config.df = df_required

# --- 4. Compose the Main Channel ---
main_label = "dynamic_channel"
main_channel = dfm.DFMIObject(
    label=main_label,
    laser_config=laser_config,
    ifo_config=main_ifo_config,
    f_samp=int(200e3) # Sampling frequency (Hz)
)
dff.sims[main_label] = main_channel

# --- 6. Simulate ---
dff.simulate(
    main_label=main_label,
    n_seconds=10, # Simulation length in seconds
    mode='snr',
    snr_db=40
)

# --- 7. Analyze and Plot ---
print("\n--- Configuration Summaries ---")
dff.sims[main_label].info()

print("\n--- Fitting Channels ---")
dff.fit(main_label)

print("\n--- Plotting Results ---")
ax = dff.plot()
ax[0].legend()
plt.show()
```
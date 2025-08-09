# DeepFMKit: A Toolkit for Deep Frequency Modulation Interferometry

**DeepFMKit** is a high-performance Python framework for the end-to-end simulation, processing, and analysis of data from Deep Frequency Modulation Interferometry (DFMI) experiments. It provides a complete computational laboratory for researchers and engineers to design virtual experiments, study systematic errors, and develop and benchmark novel data analysis techniques.

## About The Project

Deep Frequency Modulation Interferometry (DFMI) is a laser-based metrology technique used for high-precision displacement sensing and absolute distance measurement. This toolkit is designed to handle the entire experimental workflow, from simulating complex interferometric signals with realistic noise sources to performing robust, high-speed parameter estimation.

The novelty of DeepFMKit lies in its integrated, object-oriented approach. It combines a detailed physics engine—capable of modeling time-of-flight delays, colored noise sources, and arbitrary modulation waveforms—with a modular set of interchangeable fitting algorithms. This allows users to not only simulate realistic data but also to benchmark the performance of different readout schemes on identical, well-defined datasets.

### Key Features

*   **High-Fidelity Physics Engine:** Simulate realistic DFMI signals with a model that includes prescribed noise sources, dynamic path length changes, and arbitrary laser modulation waveforms.
*   **Comprehensive Noise Modeling:** Inject realistic, colored noise based on user-defined Amplitude Spectral Densities (ASDs) for laser frequency, laser amplitude, modulation amplitude, and arm length fluctuations.
*   **Modular & Interchangeable Fitters:**
    *   **NLS Fitter:** A highly optimized, parallelized Non-Linear Least Squares (NLS) fitter working in the frequency domain.
    *   **EKF Fitter:** A time-domain Extended Kalman Filter (EKF) for real-time state tracking and analysis of dynamic systems.
*   **High-Throughput Experimentation:** A declarative framework for defining and running large-scale, parallelized parameter sweeps and Monte Carlo simulations to systematically characterize system performance.
*   **Data Handling & Visualization:** A suite of tools for loading/saving data, managing experimental configurations, and plotting results.

## Getting Started

### Prerequisites

*   Python 3.9+
*   Standard scientific libraries: NumPy, SciPy, Pandas, Matplotlib
*   For high-performance noise generation: Numba

### Installation

You can either install directly from GitHub or clone the repository locally.  

**Option 1 – Install directly from GitHub (latest version):**  
```sh
pip install git+https://github.com/mdovale/DeepFMKit.git
```

**Option 2 – Clone and install manually:**  
1. **Clone this repository**
   ```sh
   git clone https://github.com/mdovale/DeepFMKit.git
   cd DeepFMKit
   ```

2. **(Recommended) Create and activate a virtual environment**
   ```sh
   python -m venv .venv
   source .venv/bin/activate     # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```


## Quick Start: A Complete Workflow

The following example demonstrates the primary workflow: defining a physical system, simulating data, running a fit, and plotting the results.

```python
import deepfmkit.core as dfm
import matplotlib.pyplot as plt

# --- 1. Define the interferometer ---
ifo = dfm.IfoConfig(
	label="main_ifo")
ifo.ref_arml = 0.1 # Reference armlength (m)
ifo.meas_arml = 0.3 # Measurement armlength (m)

# --- 2. Define the laser source ---
laser = dfm.LaserConfig(
	label="main_laser")
laser.fm = 1e3 # FM frequency (Hz)
laser.set_df_for_effect(
	ifo,
	6.54321) # Target m (rad)
	
# --- 4. Compose the main channel ---
label = "main_channel"
sim_obj = dfm.SimConfig(
    label=label,
    laser_config=laser,
    ifo_config=ifo,
    f_samp=200e3) # Acquisition frequency (Hz)

# --- 5. Instantiate the main framework ---
dff = dfm.DeepFrame(sim_obj=sim_obj)

# --- 6. Simulate ---
dff.simulate(
    label=label,
    n_seconds=10, # Simulation length (s)
    mode='snr',
    snr_db=60)

# --- 7. Print configuration ---
sim_obj.info()

# --- 8. Run readout algorithm ---
dff.fit(label)

# --- 9. Plot readout results ---
axs = dff.plot()
plt.show()
```

## Advanced Usage

### High-Fidelity Simulation with Colored Noise

The 'asd' simulation mode enables realistic noise modeling. Configure the noise properties of the LaserConfig and IfoConfig objects before simulating.

```python
from deepfmkit.physics import LaserConfig, IfoConfig, SimConfig
# Assume 'dff' and 'label' exist from the Quick Start example
# and laser/ifo variables are still in scope.

# Configure laser noise (ASD at 1 Hz, and 1/f^alpha exponent)
laser.f_n = 100.0      # 100 Hz/sqrt(Hz) of laser frequency noise
laser.f_n_alpha = 1.0  # Pink (1/f) frequency noise

# Configure interferometer noise
ifo.arml_n = 1e-12     # 1 pm/sqrt(Hz) of arm length noise
ifo.arml_n_alpha = 2.0 # Red (1/f^2) arm length noise

# Simulate using the 'asd' mode
dff.simulate(label=label, n_seconds=5, mode='asd', trial_num=42) # Use trial_num for reproducibility
```

### Using the Extended Kalman Filter (EKF)

To track a dynamic system, configure a time-varying signal and use the 'ekf' fitter.

```python
from deepfmkit.physics import LaserConfig, IfoConfig, SimConfig
# Assume 'dff', 'laser', 'ifo' exist from previous examples.

# Add a 5 Hz sinusoidal motion to the measurement arm
ifo.arml_mod_f = 5.0
ifo.arml_mod_amp = 10e-9 # 10 nm amplitude

# Re-create the SimConfig and simulate with dynamic motion
sim_dynamic = SimConfig("dynamic_channel", laser, ifo, f_samp=200e3)
dff.add_sim(sim_dynamic)
dff.simulate(label="dynamic_channel", n_seconds=2, mode='asd')

# Fit using the EKF
dff.fit(main_label="dynamic_channel", method='ekf', fit_label="ekf_fit")

# Plot the EKF's phase estimate and compare to the ground truth
ax = dff.plot(labels=['ekf_fit'], which=['phi'])
raw_data = dff.raws["dynamic_channel"]
fit_data = dff.fits["ekf_fit"]
ax.plot(fit_data.time, raw_data.phi_sim_downsamp, 'r--', label='Ground Truth')
ax.legend()
plt.show()
```

### High-Throughput Experiments

The Experiment framework (experiments.py) is designed for large-scale studies. It automates the process of running thousands of "simulate-and-analyze" trials in parallel.

1. Create an ExperimentFactory in factories.py: This class contains your logic for setting up a single trial. For example:

```python
class VairableAmplitudeOffset(ExperimentFactory):
    def __init__(self, opd_main: float = 0.1):
        self.opd_main = opd_main
        if self.opd_main == 0:
            raise ValueError("opd_main cannot be zero in the factory.")

    def _get_expected_params_keys(self) -> Set[str]:
        """Declares the top-level parameters consumed by this factory's __call__ method.
        """
        return {'m_main', 'nominal_amplitude', 'amplitude_offset'}

    def __call__(self, params: dict) -> dict:
        """Generates the physics configurations for one standard DFMI trial.
        """
        m_main = params['m_main']
        nominal_amplitude = params['nominal_amplitude']
        amplitude_offset = params['amplitude_offset']
        laser_config = physics.LaserConfig(label="ExperimentLaser")
        laser_config.amp = nominal_amplitude + amplitude_offset 
        laser_config.df = (m_main * sc.c) / (2 * np.pi * self.opd_main)
        main_ifo_config = physics.IfoConfig(label="main_ifo")
        main_ifo_config.ref_arml = 0.1
        main_ifo_config.meas_arml = main_ifo_config.ref_arml + self.opd_main
        return {
            'laser_config': laser_config,
            'main_ifo_config': main_ifo_config
        }
```

2. Define and run the Experiment:

```python
from deepfmkit.experiments import Experiment
from deepfmkit.factories import VairableAmplitudeOffset

import numpy as np
from functools import partial

def amplitude_random_offset_generator(nominal_amplitude: float, relative_noise_std: float) -> float:
    """
    Generates a random amplitude offset whose standard deviation scales
    with the `nominal_amplitude`. This simulates proportional noise.

    Parameters
    ----------
    nominal_amplitude : float
        The nominal signal amplitude, which dictates the scale of the random offset.

    Returns
    -------
    float
        A single random offset value.
    """
    noise_std = nominal_amplitude * relative_noise_std
    return np.random.normal(loc=0.0, scale=noise_std)

my_factory_instance = VairableAmplitudeOffset(
    opd_main=0.1
)

# Create an Experiment object.
experiment = Experiment(description="Example using VairableAmplitudeOffset factory")
experiment.set_config_factory(my_factory_instance)
experiment.n_fit_buffers_per_trial = 200
experiment.f_samp = 200e3

# Define sweep axis: the nominal signal amplitude.
nominal_amplitudes = np.linspace(0.5, 2.0, 10)
experiment.add_axis('nominal_amplitude', nominal_amplitudes)

# Define static parameters for all trials.
experiment.set_static({
    'm_main': 5.0,                  # Keep modulation depth constant
})

# Add stochastic variable: a random offset to the signal amplitude.
# It 'depends_on' the 'nominal_amplitude' axis.
gen_fun = partial(amplitude_random_offset_generator, relative_noise_std=0.05)

experiment.add_stochastic_variable(
    name='amplitude_offset',
    generator_func=gen_fun,
    depends_on='nominal_amplitude' # This links the noise scale to the nominal amplitude
)

# Specify the number of Monte Carlo trials for each point on the axis.
experiment.n_trials = 100 # 50 trials per nominal_amplitude point for good statistics

# Define the analysis to be performed on each simulated data set.
experiment.add_analysis(
    name='NLS_Fit',
    fitter_method='nls',
    result_cols=['amp', 'm', 'phi', 'psi', 'ssq'], # Parameters I want to collect
    fitter_kwargs={
        'n': 20,       # 20 modulation cycles per fit buffer
        'ndata': 10,   # Use this many harmonics for NLS
    }
)

# --- Run the Experiment ---
print(f"Starting experiment: '{experiment.description}'...")
experiment.results = experiment.run()
print("Experiment completed.")
```

## Citing DeepFMKit

If you use DeepFMKit in your research, please cite this paper:

```bibtex
@article{Dovale2025,
  doi = {10.48550/ARXIV.2507.23183},
  url = {https://arxiv.org/abs/2507.23183},
  author = {Dovale-\'Alvarez,  Miguel},
  keywords = {Optics (physics.optics),  Applied Physics (physics.app-ph),  Instrumentation and Detectors (physics.ins-det),  FOS: Physical sciences,  FOS: Physical sciences},
  title = {Fundamental Limitations of Absolute Ranging via Deep Frequency Modulation Interferometry},
  publisher = {arXiv},
  year = {2025},
  copyright = {arXiv.org perpetual,  non-exclusive license}
}
```

## License
This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
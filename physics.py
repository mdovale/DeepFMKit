from .data import DeepRawObject
from .plotting import default_rc

import numpy as np
import scipy.constants as sc
import pyplnoise
import pandas as pd
import logging
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
plt.rcParams.update(default_rc)
from typing import Callable, Optional, Dict, Any

class LaserConfig:
    """
    Configuration for the laser source properties in a DFMI simulation.

    This class encapsulates all parameters intrinsic to the laser, such as
    its wavelength, modulation parameters, and noise characteristics. An
    instance of this class can be shared across multiple `DFMIObject`
    instances to simulate a single laser source feeding multiple channels.

    Parameters
    ----------
    label : str, optional
        An identifier for this laser configuration.
    """
    def __init__(self, label="laser_source", psi=None):
        self.label = label
        
        # --- Core Optical Properties ---
        self.wavelength = 1.064e-6
        self.amp = 1.0
        self.visibility = 1.0
        
        # --- Modulation Properties ---
        self.f_mod = 1000
        self.df = 3e9
        self.psi = psi if psi else 0.0

        # A callable that generates the unitless modulation waveform.
        # It must accept a time axis and a phase offset as arguments.
        self.waveform_func: Callable[[np.ndarray, float], np.ndarray] = lambda t_phase: np.cos(t_phase)
        # An optional dictionary for extra arguments to the waveform function
        self.waveform_kwargs: Dict[str, Any] = {}
        
        # --- Noise Properties ---
        self.f_n = 0.0      # Frequency noise ASD (Hz/sqrt(Hz) @ 1 Hz)
        self.df_n = 0.0     # Modulation amplitude noise ASD
        self.amp_n = 0.0    # Amplitude noise ASD

    def plot_waveform(self, n_cycles=3, n_points_per_cycle=1000):
        """
        Visualizes the configured modulation waveform over a few cycles.

        This method generates and plots two key functions derived from the
        `waveform_func`:
        1. The unitless frequency modulation waveform, g(t).
        2. The resulting phase modulation waveform, phi_mod(t), scaled by df.

        This is an essential tool for verifying the behavior of custom
        waveform functions before using them in a simulation.

        Parameters
        ----------
        n_cycles : int, optional
            The number of modulation cycles to plot. Defaults to 3.
        n_points_per_cycle : int, optional
            The number of points to use for rendering each cycle, affecting
            the smoothness of the plot. Defaults to 1000.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the plot.
        """
        # --- 1. Setup Time and Phase Axes ---
        duration = n_cycles / self.f_mod
        n_total_points = n_cycles * n_points_per_cycle
        time_axis = np.linspace(0, duration, n_total_points, endpoint=False)
        omega_mod = 2 * np.pi * self.f_mod
        phase_axis = omega_mod * time_axis + self.psi

        # --- 2. Generate Waveforms using the stored function ---
        g_t = self.waveform_func(phase_axis, **self.waveform_kwargs)
        
        # I'll scale the phase modulation by df for physical units
        dt = time_axis[1] - time_axis[0]
        phi_mod = 2 * np.pi * self.df * np.cumsum(g_t) * dt
        
        # --- 3. Create the Plot ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Plot 1: Frequency Modulation
        ax1.plot(time_axis * 1000, g_t, label='g(t)')
        ax1.set_title(f"Laser Modulation Waveform: '{self.label}'")
        ax1.set_ylabel("Frequency Mod. [a.u.]")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        # Plot 2: Phase Modulation
        ax2.plot(time_axis * 1000, phi_mod, label='$\\phi_{mod}(t)$')
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Phase Mod. (rad)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        
        plt.tight_layout()
        fig.align_ylabels()
        
        return ax1, ax2
    
    def plot(self, n_cycles: int = 3, n_points_per_cycle: int = 1000, noise_seed: Optional[int] = 1):
        """
        Visualizes the full laser modulation signal and total interferometric phase generated
        by this laser, accounting for all configured noise properties (f_n, df_n, amp_n).

        This method temporarily constructs a minimal interferometer to use the
        SignalGenerator's full physics engine, allowing for a realistic
        demonstration of the laser's output under its defined noise characteristics.

        Parameters
        ----------
        n_cycles : int, optional
            The number of modulation cycles to plot. Defaults to 3.
        n_points_per_cycle : int, optional
            The number of points to use for rendering each cycle, affecting
            the smoothness of the plot. Defaults to 1000.
        noise_seed : int, optional
            Seed for the random number generator for noise, to ensure
            reproducible plots. If None, noise will be different each time.

        Returns
        -------
        tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
            The matplotlib axes objects for the two subplots. Returns (None, None)
            if `f_mod` is zero or if a necessary component is missing.
        """
        if self.f_mod == 0:
            logging.warning("Modulation frequency (f_mod) is zero. Cannot plot a full signal.")
            return None, None

        # --- 1. Setup Time Axis and Sampling Rate for Plotting ---
        duration = n_cycles / self.f_mod
        n_total_samples_plot = n_cycles * n_points_per_cycle
        time_axis = np.linspace(0, duration, n_total_samples_plot, endpoint=False)
        f_samp_plot = float(n_total_samples_plot / duration)

        # --- 2. Create a Minimal InterferometerConfig for Simulation ---
        # I create a dummy interferometer with a fixed path length difference
        # to result in a typical, non-zero modulation depth (e.g., m=15 rad)
        # using this laser's nominal df. This is purely for visualization.
        dummy_ifo_label = f"{self.label}_plot_ifo"
        dummy_ifo_config = InterferometerConfig(label=dummy_ifo_label)

        # Calculate delta_l to achieve m = 15 rad (a representative value for a plot)
        # using the nominal df of this laser.
        target_m_for_plot = 1e-6 # rad
        if self.df == 0:
            dummy_delta_l = 1e-3 # A small, arbitrary non-zero length if df is zero
            logging.warning(f"Laser modulation amplitude (df) is zero. Setting dummy delta_l to {dummy_delta_l*1e3:.1f}mm for signal visualization (m will be 0).")
        else:
            dummy_delta_l = target_m_for_plot * sc.c / (2 * np.pi * self.df)

        dummy_ifo_config.ref_arml = 0.1 # Simplest arm configuration
        dummy_ifo_config.meas_arml = dummy_delta_l

        # --- 3. Create a Dummy DFMIObject ---
        # This composes the current LaserConfig instance with the dummy InterferometerConfig.
        dummy_dfmi_channel = DFMIObject(
            label=f"{self.label}_signal_plot_channel",
            laser_config=self, # Use 'self' (this LaserConfig instance)
            ifo_config=dummy_ifo_config,
            f_samp=f_samp_plot # Use the calculated sampling rate for plotting
        )

        # --- 4. Generate Noise Arrays and Run Full Simulation Physics ---
        sg = SignalGenerator()
        
        # Generate the specific noise arrays for this plotting duration/sampling rate.
        # noise_seed is passed to ensure reproducibility.
        noise_arrays = sg._generate_noise_arrays(dummy_dfmi_channel, n_total_samples_plot, noise_seed)

        # Run the core simulation physics. is_dynamic=False as we're not simulating
        # dynamic arm motion in this context.
        _, _, witness_freq, witness_phase, _ = sg._run_simulation_physics(
            dummy_dfmi_channel, time_axis, noise_arrays, is_dynamic=False
        )

        # --- 5. Create the Plot ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot 1: Noisy Voltage Signal
        ax1.plot(time_axis * 1000, witness_freq, label='Simulated Voltage Signal $v(t)$')
        ax1.set_title(f"Simulated DFMI Signal from Laser '{self.label}' with Noise")
        ax1.set_ylabel("Voltage (a.u.)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        # Plot 2: Total Interferometric Phase (Phi_tot)
        ax2.plot(time_axis * 1000, witness_phase, label='Total Interferometric Phase $\\Phi_{\\text{tot}}(t)$')
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Phase (rad)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        
        plt.tight_layout()
        fig.align_ylabels()
        
        return ax1, ax2

class InterferometerConfig:
    """
    Configuration for the optical path of a single interferometer.

    This class encapsulates all parameters related to the interferometer's
    arm lengths and their dynamic behavior.

    Parameters
    ----------
    label : str, optional
        An identifier for this interferometer configuration.
    """
    def __init__(self, label="interferometer_path"):
        self.label = label
        
        # --- Static Path Properties ---
        self.phi = 0.0 # Interferometer phase
        self.ref_arml = 0.1 # Reference arm-length
        self.meas_arml = 0.3 # Measurement arm-length
        
        # --- Dynamic Path Properties ---
        self.arml_mod_f = 5.0 # Armlength modulation frequency
        self.arml_mod_amp = 0.0 # Armlength modulation amplitude
        self.arml_mod_psi = 0.0 # Armlength modulation phase
        self.arml_mod_n = 0.0 # Armlength modulation amplitude noise ASD

class DFMIObject:
    """
    Describes a single, complete simulation channel by composing a laser and an interferometer.

    This class acts as a container that holds a reference to a `LaserConfig`
    object and an `InterferometerConfig` object. This composite structure
    allows for a clear and physically intuitive way to define complex
    experiments, such as a W-DFMI setup where one laser feeds two different
    interferometers.

    The key derived parameter, modulation depth `m`, is provided as a read-only
    property calculated directly from the unambiguous physical parameters of the
    composed laser and interferometer objects.

    Parameters
    ----------
    label : str
        An identifier for this specific simulation channel (e.g., 'main_channel').
    laser_config : LaserConfig
        An instance of the LaserConfig class describing the light source.
    ifo_config : InterferometerConfig
        An instance of the InterferometerConfig class describing the optical path.
    f_samp : int, optional
        The sampling frequency for this channel's data acquisition.
    """
    def __init__(self, label, laser_config, ifo_config, f_samp=200000):
        self.label = label
        self.laser = laser_config # LaseConfig object
        self.ifo = ifo_config     # InterferometerConfig object
        
        # --- Simulation-specific Parameters ---
        self.f_samp = float(f_samp) # Sampling frequency (Hz)
        self.N = 0                # Number of samples, set after simulation
        self.simtime = None       # Simulation time, set after simulation
        self.fit_n = 20           # Default number of cycles to average in a fit
        self.f_fit = float(self.laser.f_mod / self.fit_n)

    @property
    def m(self):
        """
        The effective modulation index (read-only).

        This value is derived on-the-fly from the physical parameters of the
        associated LaserConfig and InterferometerConfig objects. It is not
        settable directly, which removes the ambiguity of the previous design.
        To change 'm', you must change the physical parameters `df`, `ref_arml`,
        or `meas_arml`.

        Returns
        -------
        float
            The calculated modulation depth in radians.
        """
        delta_l = self.ifo.meas_arml - self.ifo.ref_arml
        if delta_l == 0:
            return 0.0
        return 2 * np.pi * self.laser.df * delta_l / sc.c

    def info(self):
        """Prints a summary of the composed channel configuration."""
        info_str = f"""
============================================================
DFMI Channel Configuration: '{self.label}'
============================================================
--- Laser Source ('{self.laser.label}') ---
  Wavelength:        {self.laser.wavelength * 1e6:.3f} um
  Modulation Freq:   {self.laser.f_mod} Hz
  Modulation Amp (df): {self.laser.df / 1e9:.3f} GHz
  Signal Amplitude:    {self.laser.amp:.2f}
  Visibility:          {self.laser.visibility:.2f}

--- Interferometer Path ('{self.ifo.label}') ---
  Reference Arm:     {self.ifo.ref_arml:.4f} m
  Measurement Arm:   {self.ifo.meas_arml:.4f} m
  OPD (delta_l):     {(self.ifo.meas_arml - self.ifo.ref_arml)*100:.2f} cm
  Dynamic Motion Amp: {self.ifo.arml_mod_amp * 1e9:.2f} nm

--- Derived & Simulation Parameters ---
  Modulation Depth (m): {self.m:.4f} rad
  Sampling Freq:        {self.f_samp / 1e3:.1f} kHz
  Fit Cycles (fit_n):   {self.fit_n}
  Simtime / N:          {self.simtime if self.simtime else 'N/A'}{'' if self.simtime is None else ' s'} / {self.N if self.N > 0 else 'N/A'}
============================================================
"""
        logging.info(info_str)

class SignalGenerator:
    """
    A dedicated physics engine for generating DFMI time-series data.

    This class encapsulates all the logic for creating realistic DFMI signals,
    including detailed noise models and systematic effects. It supports two main
    simulation modes:
    1. 'asd': A high-fidelity mode using detailed physical noise models based
       on Amplitude Spectral Densities (ASDs).
    2. 'snr': A simplified mode that generates a perfect signal and adds a
       specified amount of white noise to achieve a target SNR.

    Its primary responsibility is to take configuration objects (`DFMIObject`)
    and produce raw data objects (`DeepRawObject`).
    """
    def generate(self, main_config, n_seconds, mode='asd', trial_num=0,
                 witness_config=None, snr_db=None, external_noise: Optional[dict] = None):
        """
        Main entry point for generating one or more linked DFMI signals.

        This method acts as a router, calling the appropriate internal simulation
        engine based on the specified mode ('asd' or 'snr'). It now supports
        the injection of pre-computed noise arrays.

        Parameters
        ----------
        main_config : DFMIObject
            The configuration for the primary measurement channel.
        n_seconds : float
            The duration of the time series to generate in seconds.
        mode : {'asd', 'snr'}, optional
            The simulation mode to use. Defaults to 'asd'.
        trial_num : int, optional
            A number used to seed the random noise generators.
        witness_config : DFMIObject, optional
            The configuration for a secondary witness channel.
        snr_db : float, optional
            The target Signal-to-Noise Ratio in dB. Required if mode='snr'.
        external_noise : dict, optional
            A dictionary of pre-computed noise time-series arrays. If provided,
            the internal noise generation is skipped, and these arrays are used
            instead. The keys must match the noise source names (e.g.,
            'laser_frequency', 'amplitude'). Defaults to None.

        Returns
        -------
        dict
            A dictionary of DeepRawObject instances keyed by 'main' and 'witness'.
        """
        if mode == 'asd':
            return self._generate_with_asd(main_config, n_seconds, trial_num, witness_config, external_noise)
        elif mode == 'snr':
            if snr_db is None:
                logging.error("SNR mode requires a value for 'snr_db'.")
                return {}
            # Note: external_noise is ignored in 'snr' mode for simplicity.
            return self._generate_with_snr(main_config, n_seconds, trial_num, snr_db)
        else:
            logging.error(f"Unknown simulation mode: '{mode}'")
            return {}

    def _generate_with_asd(self, main_config, n_seconds, trial_num, witness_config, external_noise=None):
        """Generates a signal using detailed physical noise ASDs."""
        num_samples = int(n_seconds * main_config.f_samp)
        time_axis = np.arange(num_samples) / main_config.f_samp
        main_config.N = len(time_axis)

        # If external noise is provided, use it. Otherwise, generate it internally.
        if external_noise:
            logging.debug("Using externally provided noise arrays.")
            # Ensure all required noise keys exist, even if they are zero.
            noise_keys = ['laser_frequency', 'amplitude', 'df', 'armlength']
            noise = {key: external_noise.get(key, 0.0) for key in noise_keys}
        else:
            logging.debug("Generating noise internally based on ASDs.")
            noise = self._generate_noise_arrays(main_config, len(time_axis), trial_num)
        
        # The physics engine now returns the ground truth phase
        dfmi_signal, dfmi_phase, _, _, simulated_phase_ground_truth = self._run_simulation_physics(main_config, time_axis, noise, is_dynamic=True)

        raw_main = DeepRawObject(data=pd.DataFrame(dfmi_signal, columns=["ch0"]))
        raw_main.label = main_config.label
        raw_main.f_samp = main_config.f_samp
        raw_main.f_mod = main_config.laser.f_mod
        raw_main.sim = main_config
        
        # Store all ground truth and noise signals
        raw_main.phi = dfmi_phase # The full phase with all noise
        raw_main.phi_sim = simulated_phase_ground_truth # The ground truth phase to recover
        raw_main.f_noise = noise.get('laser_frequency', 0.0)
        raw_main.a_noise = noise.get('amplitude', 0.0)
        raw_main.l_noise = noise.get('armlength', 0.0)
        raw_main.df_noise = noise.get('df', 0.0)

        output_channels = {'main': raw_main}

        if witness_config is not None:
            # Witness uses the same common noise but its own (static) physics
            witness_signal, witness_phase, _ , _, simulated_phase_ground_truth = self._run_simulation_physics(witness_config, time_axis, noise, is_dynamic=False)
            raw_witness = DeepRawObject(data=pd.DataFrame(witness_signal, columns=["ch0"]))
            raw_witness.label = witness_config.label
            raw_witness.f_samp = witness_config.f_samp
            raw_witness.f_mod = witness_config.laser.f_mod
            raw_witness.sim = witness_config
            # Store its ground truth and common noise
            raw_witness.phi = witness_phase
            raw_witness.phi_sim = simulated_phase_ground_truth
            raw_witness.f_noise = noise.get('laser_frequency', 0.0)
            raw_witness.a_noise = noise.get('amplitude', 0.0)
            output_channels['witness'] = raw_witness

        return output_channels

    def _generate_with_snr(self, main_config, n_seconds, trial_num, snr_db):
        """Generates a signal with a specific SNR."""
        num_samples = int(n_seconds * main_config.f_samp)
        time_axis = np.arange(num_samples) / main_config.f_samp
        main_config.N = len(time_axis)
        
        y_clean = self._generate_ideal_signal(main_config, time_axis, is_dynamic=False)
        y_noisy = self._add_white_noise(y_clean, snr_db, trial_num)
        
        raw_obj = DeepRawObject(data=pd.DataFrame(y_noisy, columns=["ch0"]))
        raw_obj.label = main_config.label
        raw_obj.f_samp = main_config.f_samp
        raw_obj.f_mod = main_config.laser.f_mod
        raw_obj.t0 = 0
        raw_obj.sim = main_config
        
        return {'main': raw_obj}
    
    def _generate_ideal_signal(self, sim_config, time_axis, is_dynamic):
        """Generates a perfect, noiseless DFMI signal."""
        laser = sim_config.laser
        ifo = sim_config.ifo

        A = laser.amp
        C = laser.visibility
        omega_mod = 2 * np.pi * laser.f_mod
        omega_0 = 2 * np.pi * sc.c / laser.wavelength

        tau_r = ifo.ref_arml / sc.c
        tau_m = ifo.meas_arml / sc.c

        if is_dynamic:
            tau_dl = (0.5 * ifo.arml_mod_amp * np.sin(2 * np.pi * ifo.arml_mod_f * time_axis + ifo.arml_mod_psi)
                      + ifo.phi * laser.wavelength / (2 * np.pi)) / sc.c
            sin_term_meas = np.sin(omega_mod * (time_axis - tau_m - tau_dl) + laser.psi)
            sin_term_ref = np.sin(omega_mod * (time_axis - tau_r) + laser.psi)
            phi_static = omega_0 * (tau_m - tau_r + tau_dl)
            phi_mod = (laser.df / laser.f_mod) * (sin_term_meas - sin_term_ref)
            phitot = phi_static + phi_mod
        else:
            m = sim_config.m # Use the property for convenience
            phitot = ifo.phi + m * np.cos(omega_mod * time_axis + laser.psi)

        return A * (1 + C * np.cos(phitot))

    def _add_white_noise(self, clean_signal, snr_db, trial_num):
        """Adds white Gaussian noise to a signal to achieve a target SNR."""
        signal_ac = clean_signal - np.mean(clean_signal)
        signal_power = np.mean(signal_ac**2)
        snr_linear_power = 10**(snr_db / 10.0)
        noise_power = signal_power / snr_linear_power
        noise_std_dev = np.sqrt(noise_power)

        rng = np.random.RandomState(seed=trial_num)
        noise = rng.randn(len(clean_signal)) * noise_std_dev
        return clean_signal + noise

    def _generate_noise_arrays(self, sim_config, n_samples, trial_num=0):
        """
        Generates time-series noise arrays for different physical sources.

        This function creates noise based on the Amplitude Spectral Density (ASD)
        and noise color (`alpha`) defined in the simulation configuration.

        It implements special, optimized handling for white noise sources
        ('amplitude', 'df'), using `numpy.random.normal`. The required standard
        deviation (`sigma`) for the time series is calculated from the target ASD
        using the relation: `sigma = ASD * sqrt(sampling_rate / 2)`.

        For other, colored noise sources (like 1/f laser frequency drift), it
        continues to use the `pyplnoise` library to generate the appropriate
        time series.

        Parameters
        ----------
        sim_config : DFMIObject
            The simulation configuration object containing noise parameters like
            `f_n`, `amp_n`, `df_n`, etc.
        n_samples : int
            Number of data samples to generate
        trial_num : int, optional
            A number used to seed the random noise generators, ensuring
            reproducible but different noise for each trial. Defaults to 0.

        Returns
        -------
        dict
            A dictionary where keys are noise source names (e.g., 'laser_frequency',
            'amplitude') and values are the corresponding numpy arrays of the
            generated noise time series.
        """
        num_samples = int(n_samples)
        assert num_samples > 0, f"num_samples should be greater than zero, got {num_samples}"
        fs = sim_config.f_samp
        noise_params = {
            'laser_frequency': {'asd': sim_config.laser.f_n, 'alpha': 2.0},
            'amplitude': {'asd': sim_config.laser.amp_n, 'alpha': 0.0},
            'df': {'asd': sim_config.laser.df_n, 'alpha': 0.0},
            'armlength': {'asd': sim_config.ifo.arml_mod_n, 'alpha': 2.0}
        }
        basis_noises = {}
        final_noise = {}
        
        # Use a single seed counter to ensure all generators are unique per trial
        seed_counter = 1 + trial_num * len(noise_params)

        # Create a single RandomState generator for all numpy-based noise
        rng = np.random.RandomState(seed=seed_counter)
        seed_counter += 1

        for name, params in noise_params.items():
            asd = params['asd']
            if asd == 0.0:
                final_noise[name] = 0.0
                continue

            if name in ['amplitude', 'df']:
                # Calculate required standard deviation for the time series
                # sigma = ASD * sqrt(sampling_rate / 2)
                sigma = asd * np.sqrt(fs / 2.0)
                final_noise[name] = rng.normal(scale=sigma, size=num_samples)
                # Skip to the next noise source
                continue

            alpha_val = params['alpha']
            # Generate basis noise if not already created for this color
            if alpha_val not in basis_noises and params['asd'] != 0:
                generator = pyplnoise.AlphaNoise(fs, fs / num_samples, fs / 2, 
                                                alpha=alpha_val, seed=seed_counter)
                basis_noises[alpha_val] = generator.get_series(num_samples)
                seed_counter += 1
            
            # Scale the basis noise by the specified ASD
            if alpha_val in basis_noises:
                final_noise[name] = asd / np.sqrt(2) * basis_noises[alpha_val]
            else:
                final_noise[name] = 0.0

        return final_noise

    def _run_simulation_physics(self, sim_config: DFMIObject, time_axis: np.ndarray, noise_arrays: Dict[str, np.ndarray], is_dynamic: bool = True):
        """
        Core physics engine for a single channel using the EXACT physical model.

        This function calculates the final time-series voltage of a DFMI signal.
        It uses the exact physical model for the phase modulation by integrating the
        true frequency waveform and then calculating the phase difference
        `phi_mod(t-tau) - phi_mod(t)`. This ensures high fidelity, especially
        for large modulation depths.

        Parameters
        ----------
        sim_config : DFMIObject
            The composed configuration object for the channel.
        time_axis : numpy.ndarray
            The time vector for the simulation.
        noise_arrays : dict
            A dictionary containing the pre-generated time series for each noise source.
            Expected keys: 'df' (modulation amplitude noise), 'laser_frequency' (frequency noise),
            'amplitude' (amplitude noise on overall signal amplitude), 'armlength' (arm length noise).
        is_dynamic : bool, optional
            If True, includes dynamic arm length modulation and noise.

        Returns
        -------
        dfmi_signal : numpy.ndarray
            The final simulated DFMI voltage time series.
        dfmi_phase : numpy.ndarray
            The total interferometric phase within the cosine term of the DFMI voltage time series.
        witness_signal : numpy.ndarray
            Voltage signal "witnessed" by idealized heterodyne witness interferometer,
            capturing the laser's instantaneous frequency modulation and noise.
        witness_phase : numpy.ndarray
            The instantaneous phase of the ideal heterodyne witness interferometer's beatnote.
        simulated_phase_ground_truth : numpy.ndarray
            The ground truth interferometric phase from path length changes only,
            excluding frequency modulation and laser frequency noise.
        """
        # --- Unpack Configuration and Construct True Frequency Waveform ---
        laser = sim_config.laser
        ifo = sim_config.ifo
        omega_mod = 2 * np.pi * laser.f_mod

        # --- Construct the normalized, unitless frequency modulation waveform g(t) ---
        # by calling the function stored in the laser config.
        phase_axis = omega_mod * time_axis + laser.psi
        g_t = laser.waveform_func(phase_axis, **laser.waveform_kwargs)

        # Ensure g_t is properly normalized, handling potential division by zero if g_t is all zeros
        max_abs_g_t = np.max(np.abs(g_t))
        g_t_normalized = g_t / max_abs_g_t if max_abs_g_t != 0 else np.zeros_like(g_t)
        
        # --- Construct True Phase Modulation Waveform phi_mod(t) ---
        # Integrate the true frequency waveform to get the phase waveform
        df_noisy = laser.df + noise_arrays.get('df', 0.0)

        dt = time_axis[1] - time_axis[0]
        fs = 1/dt
        phi_mod_waveform = (2*np.pi/fs)*np.cumsum(df_noisy * g_t_normalized)

        # Using cumtrapz for better accuracy in numerical integration
        # phi_mod_waveform = 2 * np.pi * cumulative_trapezoid(df_noisy * g_t_normalized, x=time_axis, initial=0.0)
        
        # --- Calculate Delays and Path Lengths ---
        tau_r = ifo.ref_arml / sc.c
        tau_m = ifo.meas_arml / sc.c
        
        if not is_dynamic:
            path_length_change = ifo.phi * laser.wavelength / (2 * np.pi)
        else:
            path_length_change = (ifo.arml_mod_amp * np.sin(2 * np.pi * ifo.arml_mod_f * time_axis + ifo.arml_mod_psi)
                                + noise_arrays.get('armlength', 0.0)
                                + ifo.phi * laser.wavelength / (2 * np.pi))
        tau_dl = path_length_change / sc.c

        # --- Calculate Final Phase using the EXACT Model ---
        t_interp_meas = time_axis - (tau_m + tau_dl)
        t_interp_ref = time_axis - tau_r
        
        # --- Interpolate the full phase modulation waveform at delayed times ---
        phi_mod_meas = np.interp(t_interp_meas, time_axis, phi_mod_waveform)
        phi_mod_ref = np.interp(t_interp_ref, time_axis, phi_mod_waveform)
        delta_phi_mod = phi_mod_meas - phi_mod_ref
        
        # --- Add the carrier phase term, including frequency noise ---
        f0_noisy = (sc.c / laser.wavelength) + noise_arrays.get('laser_frequency', 0.0)
        omega_0_noisy = 2 * np.pi * f0_noisy
        delta_phi_carrier = omega_0_noisy * ((tau_m + tau_dl) - tau_r)

        dfmi_phase = delta_phi_carrier + delta_phi_mod
        
        # --- Generate Final DFMI Signal and Ground Truth ---
        # Calculate effective amplitude, incorporating amplitude noise multiplicatively
        amplitude_effective = laser.amp + noise_arrays.get('amplitude', 0.0)
        
        # The DFMI signal: A_effective * [1 + k * cos(Phi)]
        dfmi_signal = amplitude_effective * (1 + laser.visibility * np.cos(dfmi_phase))
        
        omega_0_clean = 2 * np.pi * sc.c / laser.wavelength
        simulated_phase_ground_truth = omega_0_clean * ((tau_m + tau_dl) - tau_r)
        if np.isscalar(simulated_phase_ground_truth):
            simulated_phase_ground_truth = np.full_like(time_axis, simulated_phase_ground_truth, dtype=float)

        # --- Generate the Laser's True (Noisy) "Heterodyne-Witness" Signal ---
        witness_freq = (df_noisy * g_t_normalized) + noise_arrays.get('laser_frequency', 0.0)
        witness_phase = (2*np.pi/fs)*np.cumsum(np.array(witness_freq-np.mean(witness_freq)))
        
        return dfmi_signal, dfmi_phase, witness_freq, witness_phase, simulated_phase_ground_truth
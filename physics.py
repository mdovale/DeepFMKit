import numpy as np
import scipy.constants as sc
import pyplnoise
import pandas as pd
import logging
from .data import DeepRawObject 

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
    def __init__(self, label="laser_source"):
        self.label = label
        
        # --- Core Optical Properties ---
        self.wavelength = 1.064e-6
        self.amp = 1.0
        self.visibility = 1.0
        
        # --- Modulation Properties ---
        self.f_mod = 1000
        self.df = 3e9
        self.psi = 0.0
        
        # --- Noise Properties ---
        self.f_n = 0.0      # Frequency noise ASD (Hz/sqrt(Hz) @ 1 Hz)
        self.df_n = 0.0     # Modulation amplitude noise ASD
        self.amp_n = 0.0    # Amplitude noise ASD
        
        # --- Systematic Error Properties ---
        self.df_2nd_harmonic_frac = 0.0
        self.df_2nd_harmonic_phase = 0.0

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
        self.phi = 0.0
        self.ref_arml = 0.1
        self.meas_arml = 0.3
        
        # --- Dynamic Path Properties ---
        self.arml_mod_f = 5.0
        self.arml_mod_amp = 0.0
        self.arml_mod_psi = 0.0
        self.arml_mod_n = 0.0 # Armlength noise ASD

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
        self.f_samp = int(f_samp) # Sampling frequency (Hz)
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
                 witness_config=None, snr_db=None):
        """
        Main entry point for generating one or more linked DFMI signals.

        This method acts as a router, calling the appropriate internal simulation
        engine based on the specified mode ('asd' or 'snr').

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

        Returns
        -------
        dict
            A dictionary of DeepRawObject instances. It will always contain
            a 'main' key. If a witness is generated, it will also contain
            a 'witness' key.
        """
        if mode == 'asd':
            return self._generate_with_asd(main_config, n_seconds, trial_num, witness_config)
        elif mode == 'snr':
            if snr_db is None:
                logging.error("SNR mode requires a value for 'snr_db'.")
                return {}
            return self._generate_with_snr(main_config, n_seconds, trial_num, snr_db)
        else:
            logging.error(f"Unknown simulation mode: '{mode}'")
            return {}

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
        raw_obj.f_mod = main_config.f_mod
        raw_obj.t0 = 0
        raw_obj.sim = main_config
        
        return {'main': raw_obj}

    def _generate_with_asd(self, main_config, n_seconds, trial_num, witness_config):
        """Generates a signal using detailed physical noise ASDs."""
        num_samples = int(n_seconds * main_config.f_samp)
        time_axis = np.arange(num_samples) / main_config.f_samp
        main_config.N = len(time_axis)

        # Noise generation is based on the single shared laser and the main IFO's motion
        noise = self._generate_noise_arrays(main_config, time_axis, trial_num)
        
        # The physics engine now returns the ground truth phase
        y_main, phitot_main, phi_sim_main = self._run_simulation_physics(main_config, time_axis, noise, is_dynamic=True)

        raw_main = DeepRawObject(data=pd.DataFrame(y_main, columns=["ch0"]))
        raw_main.label = main_config.label
        raw_main.f_samp = main_config.f_samp
        raw_main.f_mod = main_config.laser.f_mod
        raw_main.sim = main_config
        
        # --- Store all ground truth and noise signals ---
        raw_main.phi = phitot_main # The full phase with all noise
        raw_main.phi_sim = phi_sim_main # The ground truth phase to recover
        raw_main.f_noise = noise.get('laser_frequency', 0.0)
        raw_main.a_noise = noise.get('amplitude', 0.0)
        raw_main.l_noise = noise.get('armlength', 0.0)
        raw_main.df_noise = noise.get('df', 0.0)

        output_channels = {'main': raw_main}

        if witness_config is not None:
            # Witness uses the same common noise but its own (static) physics
            y_witness, phitot_witness, phi_sim_witness = self._run_simulation_physics(witness_config, time_axis, noise, is_dynamic=False)
            raw_witness = DeepRawObject(data=pd.DataFrame(y_witness, columns=["ch0"]))
            raw_witness.label = witness_config.label
            raw_witness.f_samp = witness_config.f_samp
            raw_witness.f_mod = witness_config.laser.f_mod
            raw_witness.sim = witness_config
            # Store its ground truth and common noise
            raw_witness.phi = phitot_witness
            raw_witness.phi_sim = phi_sim_witness
            raw_witness.f_noise = noise.get('laser_frequency', 0.0)
            raw_witness.a_noise = noise.get('amplitude', 0.0)
            output_channels['witness'] = raw_witness

        return output_channels
    
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

    def _generate_noise_arrays(self, sim_config, time_axis, trial_num=0):
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
        time_axis : numpy.ndarray
            The time vector for the simulation. Its length determines the number
            of samples to generate.
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
        num_samples = len(time_axis)
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

    def _run_simulation_physics(self, sim_config, time_axis, noise_arrays, is_dynamic=True):
        """
        Core physics engine for a single channel with detailed noise models.

        This function calculates the final time-series voltage of a DFMI signal
        by combining the laser properties, interferometer path information, and
        various stochastic noise sources.

        It also calculates and returns the "ground truth" interferometric phase,
        `phi_sim`, which represents the ideal phase evolution from physical path
        length changes, excluding laser frequency noise and the DFMI modulation itself.
        This method ensures `phi_sim` is always returned as a NumPy array.

        Parameters
        ----------
        sim_config : DFMIObject
            The composed configuration object for the channel.
        time_axis : numpy.ndarray
            The time vector for the simulation.
        noise_arrays : dict
            A dictionary containing the pre-generated time series for each noise source.
        is_dynamic : bool, optional
            If True, includes dynamic arm length modulation and noise. If False,
            simulates a static interferometer. Defaults to True.

        Returns
        -------
        y_final : numpy.ndarray
            The final simulated voltage time series.
        phitot_final : numpy.ndarray
            The total phase inside the final cosine term, including all noise
            and modulation effects.
        phi_sim_ground_truth : numpy.ndarray
            The ground truth interferometric phase from path length changes only.
            This is the ideal signal for validation.
        """
        # --- 1. Unpack configuration and noisy terms ---
        laser = sim_config.laser
        ifo = sim_config.ifo

        A = laser.amp
        C = laser.visibility
        omega_0_noisy = 2 * np.pi * (sc.c / laser.wavelength + noise_arrays.get('laser_frequency', 0.0))
        omega_mod = 2 * np.pi * laser.f_mod
        df_noisy = laser.df + noise_arrays.get('df', 0.0)

        tau_r = ifo.ref_arml / sc.c
        tau_m = ifo.meas_arml / sc.c

        # --- 2. Calculate time-varying path length component, `tau_dl` ---
        if not is_dynamic:
            # For static channels, path length change is constant from phi offset
            path_length_change = ifo.phi * laser.wavelength / (2 * np.pi)
        else:
            # For dynamic channels, include motion and noise
            path_length_change = (ifo.arml_mod_amp * np.sin(2 * np.pi * ifo.arml_mod_f * time_axis + ifo.arml_mod_psi)
                                + noise_arrays.get('armlength', 0.0)
                                + ifo.phi * laser.wavelength / (2 * np.pi))
        
        # The total time delay change
        tau_dl = path_length_change / sc.c

        # --- 3. Calculate full DFMI phase ---
        sin_term_meas = np.sin(omega_mod * (time_axis - tau_m - tau_dl) + laser.psi)
        sin_term_ref = np.sin(omega_mod * (time_axis - tau_r) + laser.psi)
        phi_mod = (df_noisy / laser.f_mod) * (sin_term_meas - sin_term_ref)

        if laser.df_2nd_harmonic_frac != 0:
            eps = laser.df_2nd_harmonic_frac
            theta_eps = laser.df_2nd_harmonic_phase
            sin_term_meas_2nd = np.sin(2 * omega_mod * (time_axis - tau_m - tau_dl) + laser.psi + theta_eps)
            sin_term_ref_2nd = np.sin(2 * omega_mod * (time_axis - tau_r) + laser.psi + theta_eps)
            phi_mod += (df_noisy * eps / (2 * laser.f_mod)) * (sin_term_meas_2nd - sin_term_ref_2nd)

        phi_static_and_drift = omega_0_noisy * (tau_m - tau_r + tau_dl)
        phitot_final = phi_static_and_drift + phi_mod

        # --- 4. Calculate final signal and ground truth phase (`phi_sim`) ---
        y_final = A + noise_arrays.get('amplitude', 0.0) + A * C * np.cos(phitot_final)
        
        # --- BUG FIX: Ensure `phi_sim` is always an array of the correct length ---
        omega_0_clean = 2 * np.pi * sc.c / laser.wavelength
        
        # Calculate the ground truth path length delay. This is an array for dynamic
        # cases and a float for static cases.
        tau_dl_ground_truth = path_length_change / sc.c
        
        # Ensure the final result is broadcast to the shape of time_axis
        phi_sim_ground_truth = omega_0_clean * (tau_m - tau_r + tau_dl_ground_truth)
        
        # If the result is a scalar, broadcast it to a full array.
        if np.isscalar(phi_sim_ground_truth):
            phi_sim_ground_truth = np.full_like(time_axis, phi_sim_ground_truth, dtype=float)

        return y_final, phitot_final, phi_sim_ground_truth
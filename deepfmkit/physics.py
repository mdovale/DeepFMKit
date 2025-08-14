# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
from deepfmkit.data import RawData
from deepfmkit.plotting import plot_signal_harmonics, plot_signal_harmonics_vs_phi
from deepfmkit.noise import alpha_noise
from deepfmkit.dsp import timeshift
from deepfmkit import waveforms

import math
import numpy as np
import scipy.constants as sc
import pandas as pd
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid, quad
from scipy.signal import decimate
from typing import Callable, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import matplotlib.axes

# --- Helpers ---
def is_valid_alpha(alpha: float) -> bool:
    """Return True iff alpha is finite and either 0.0 or in [0.01, 2.0]."""
    return math.isfinite(alpha) and (alpha == 0.0 or 0.01 <= alpha <= 2.0)

def validate_alpha(alpha: float, name: str) -> None:
    """Validate a single alpha; raise with a consistent message."""
    if not is_valid_alpha(alpha):
        raise ValueError(
            f"{name} must be finite and either 0.0 or in the closed interval [0.01, 2.0]."
        )

@dataclass
class IfoConfig:
    """Interferometer configuration.

    This class encapsulates all parameters related to an interferometer's
    arm lengths, their dynamic behavior, and associated noise.

    Attributes
    ----------
    label : str
        An identifier for this interferometer configuration (e.g., "main_ifo").
    visibility : float
        The visibility (contrast) of the interference fringes, from 0 to 1.
    phi : float
        The static interferometric phase offset in radians. This represents
        the phase difference at the carrier frequency for the static part
        of the OPD.
    ref_arml : float
        The one-way optical path length of the reference arm in meters.
    meas_arml : float
        The one-way optical path length of the measurement arm in meters.
    arml_mod_f : float
        The frequency of any prescribed sinusoidal arm length modulation in Hz.
    arml_mod_amp : float
        The amplitude of the prescribed sinusoidal arm length modulation in meters.
    arml_mod_psi : float
        The phase of the prescribed sinusoidal arm length modulation in radians.
    arml_n : float
        The Amplitude Spectral Density (ASD) of arm length noise at 1 Hz,
        in units of m/sqrt(Hz).
    arml_n_alpha : float
        The exponent `alpha` for the 1/f^alpha power spectrum of the arm
        length noise.
    s_n : float
        The Amplitude Spectral Density (ASD) of electronic noise at 1 Hz,
        in units of V/sqrt(Hz).
    s_n_alpha : float
        The exponent `alpha` for the 1/f^alpha power spectrum of the
        electronic noise.
    """

    label: str = "IFO"
    visibility: float = 1.0

    # --- Static Path Properties ---
    phi: float = 0.0  # Interferometer phase
    ref_arml: float = 0.1  # Reference optical arm-length
    meas_arml: float = 0.3  # Measurement optical arm-length

    # --- Dynamic Path Properties ---
    arml_mod_f: float = 5.0  # Sinusoidal modulation frequency
    arml_mod_amp: float = 0.0  # Modulation amplitude
    arml_mod_psi: float = 0.0  # Modulation phase

    # --- Noise Properties ---
    arml_n: float = 0.0  # ASD of armlength noise (m/sqrt(Hz) @ 1 Hz)
    arml_n_alpha: float = 2.0  # Color of armlength noise (PSD \propto 1/f^alpha)
    s_n: float = 0.0  # ASD of electronic noise (m/sqrt(Hz) @ 1 Hz)
    s_n_alpha: float = 0.0 # Color of electronic noise (PSD \propto 1/f^alpha)

    def __post_init__(self):
        """Validates the configuration after initialization."""
        if not (0.0 <= self.visibility <= 1.0):
            raise ValueError("visibility must be between 0.0 and 1.0 inclusive.")
        
        # --- Static Path Properties ---
        if not math.isfinite(self.phi):
            raise ValueError("phi must be a finite float (radians).")
        if self.ref_arml <= 0.0 or not math.isfinite(self.ref_arml):
            raise ValueError("ref_arml must be a finite, positive length (meters).")
        if self.meas_arml <= 0.0 or not math.isfinite(self.meas_arml):
            raise ValueError("meas_arml must be a finite, positive length (meters).")

        # --- Dynamic Path Properties ---
        if self.arml_mod_f < 0.0 or not math.isfinite(self.arml_mod_f):
            raise ValueError("arml_mod_f must be finite and non-negative (Hz).")
        if self.arml_mod_amp < 0.0 or not math.isfinite(self.arml_mod_amp):
            raise ValueError("arml_mod_amp must be finite and non-negative (meters).")
        if not math.isfinite(self.arml_mod_psi):
            raise ValueError("arml_mod_psi must be a finite float (radians).")

        # --- Noise Properties ---
        if self.arml_n < 0.0 or not math.isfinite(self.arml_n):
            raise ValueError("arml_n must be finite and non-negative (m/sqrt(Hz)).")
        validate_alpha(self.arml_n_alpha, "arml_n_alpha")

        if self.s_n < 0.0 or not math.isfinite(self.s_n):
            raise ValueError("s_n must be finite and non-negative (m/sqrt(Hz)).")
        validate_alpha(self.s_n_alpha, "s_n_alpha")


@dataclass
class LaserConfig:
    """Configuration for the laser source.

    This class encapsulates all parameters intrinsic to the laser, such as its
    wavelength, modulation properties, and noise characteristics. An instance
    can be shared across multiple `SimConfig` objects to simulate a single
    laser feeding multiple interferometers.

    Attributes
    ----------
    label : str
        An identifier for this laser configuration.
    wavelength : float
        The central vacuum wavelength of the laser in meters.
    amp : float
        The mean amplitude of the interferometric signal in volts. This is an
        effective parameter combining laser power, photodetector gain, etc.
    fm : float
        The frequency of the laser's frequency modulation in Hz.
    df : float
        The amplitude (peak deviation) of the laser's frequency modulation
        in Hz.
    psi : float
        The phase offset of the laser's frequency modulation in radians.
    waveform_func : Callable
        A Python function that generates the unitless modulation waveform.
        It must accept a phase axis (`omega_m * t + psi`) as its first
        argument and can accept additional keyword arguments.
    waveform_kwargs : dict
        A dictionary of keyword arguments to pass to `waveform_func`.
    f_n : float
        ASD of laser frequency noise at 1 Hz, in Hz/sqrt(Hz).
    df_n : float
        ASD of modulation amplitude noise at 1 Hz, in Hz/sqrt(Hz).
    r_n : float
        ASD of signal amplitude noise at 1 Hz, in V/sqrt(Hz).
    f_n_alpha, df_n_alpha, r_n_alpha : float
        The exponents `alpha` for the 1/f^alpha power spectra of the
        corresponding noise sources.
    """

    label: str = "Laser"

    # --- Core Optical Properties ---
    wavelength: float = 1.064e-6
    amp: float = 1.0

    # --- Modulation Properties ---
    fm: float = 1000.0
    df: float = 3e9
    psi: float = 0.0

    # A callable that generates the unitless modulation waveform
    waveform_func: Callable[..., np.ndarray] = waveforms.cosine

    # An optional dictionary for extra arguments to the waveform function
    waveform_kwargs: Dict[str, Any] = field(default_factory=dict)

    # --- Noise Properties ---
    # Noise levels:
    f_n: float = 0.0  # ASD of frequency noise (Hz/sqrt(Hz) @ 1 Hz)
    df_n: float = 0.0  # ASD of modulation amplitude noise (Hz/sqrt(Hz) @ 1 Hz)
    r_n: float = 0.0  # ASD of relative intensity noise (V/sqrt(Hz) @ 1 Hz)
    # Color of noise (PSD \propto 1/f^alpha):
    f_n_alpha: float = 2.0  # Color of frequency noise
    df_n_alpha: float = 0.0  # Color of modulation amplitude noise
    r_n_alpha: float = 0.0  # Color of relative intensity noise

    def __post_init__(self):
        """Validates the configuration after initialization."""
        if self.wavelength <= 0:
            raise ValueError("wavelength must be > 0 (meters).")
        if self.fm <= 0:
            raise ValueError("modulation frequency must be >0 (Hz).")
        if self.df <= 0:
            raise ValueError("modulation amplitude must be >0 (Hz).")
        if not callable(self.waveform_func):
            raise TypeError("waveform_func must be callable.")

        # noise ASDs (must be finite and ≥ 0)
        for name, val in (("f_n", self.f_n), ("df_n", self.df_n), ("r_n", self.r_n)):
            if not (math.isfinite(val) and val >= 0.0):
                raise ValueError(f"{name} must be a finite, non-negative ASD.")

        # noise colors: each alpha must be finite and either 0.0 or in [0.01, 2.0]
        alphas = {
            "f_n_alpha": self.f_n_alpha,
            "df_n_alpha": self.df_n_alpha,
            "r_n_alpha": self.r_n_alpha,
        }
        bad = [k for k, v in alphas.items() if not is_valid_alpha(v)]
        if bad:
            raise ValueError(
                "Noise PSD is proportional to 1/f^alpha; each alpha must be finite and either 0.0 or in the closed interval [0.01, 2.0]. "
                f"Out of range: {', '.join(bad)}."
            )

    def set_df_for_effect(self, ifo: IfoConfig, m: float) -> None:
        """Sets laser's `df` to achieve a target modulation index `m`.

        This convenience method calculates the required laser frequency
        modulation amplitude (`df`) to produce a desired effective modulation
        index (`m`) in a specific interferometer.

        Parameters
        ----------
        ifo : IfoConfig
            The target interferometer configuration, used to determine the
            optical path difference.
        m : float
            The target effective modulation index in radians.
        """
        opd = abs(ifo.meas_arml - ifo.ref_arml)
        if opd == 0:
            self.df = 0.0
            logging.warning(f"IFO '{ifo.label}' has zero OPD. Setting df=0.")
            return
        self.df = (m * sc.c) / (2 * np.pi * opd)

    def set_df_for_m(self, ifo: IfoConfig, m: float) -> None:
        """See set_df_for_effect"""
        self.set_df_for_effect(ifo, m)

    def plot_waveform(
        self, n_cycles: int = 3, n_points_per_cycle: int = 1000
    ) -> Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]:
        """Visualizes the configured laser modulation waveform.

        This method plots the unitless frequency modulation waveform g(t)
        and the resulting physical phase modulation phi_mod(t). It is a
        useful tool for verifying the behavior of custom waveform functions.

        Parameters
        ----------
        n_cycles : int, optional
            The number of modulation cycles to plot. Defaults to 3.
        n_points_per_cycle : int, optional
            The number of points for rendering each cycle, affecting plot
            smoothness. Defaults to 1000.

        Returns
        -------
        tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
            A tuple containing the two matplotlib Axes objects for the plots.
        """
        # --- 1. Setup time and phase Axes ---
        duration = n_cycles / self.fm
        n_total_points = n_cycles * n_points_per_cycle
        time_axis = np.linspace(0, duration, n_total_points, endpoint=False)
        omega_mod = 2 * np.pi * self.fm
        phase_axis = omega_mod * time_axis + self.psi

        # --- 2. Generate Waveforms using the stored function ---
        g_t = self.waveform_func(phase_axis, **self.waveform_kwargs)

        # Scale the phase modulation by df for physical units
        dt = time_axis[1] - time_axis[0]
        phi_mod = 2 * np.pi * self.df * np.cumsum(g_t) * dt

        # --- 3. Create the Plot ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Plot 1: Frequency modulation
        ax1.plot(time_axis * 1000, g_t, label="g(t)")
        ax1.set_title(f"Laser Modulation Waveform: '{self.label}'")
        ax1.set_ylabel("Frequency Mod. [a.u.]")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # Plot 2: Phase modulation
        ax2.plot(time_axis * 1000, phi_mod, label="$\\phi_{mod}(t)$")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Phase Mod. (rad)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        fig.align_ylabels()

        return ax1, ax2

    def plot(
        self,
        n_cycles: int = 3,
        n_points_per_cycle: int = 1000,
        noise_seed: Optional[int] = 1,
    ) -> Optional[Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]:
        """Visualizes the full simulated DFMI signal from this laser on a
        near equal-arm interferometer.

        This method runs a minimal but complete physics simulation to show
        the final voltage and total phase signal produced by this laser,
        including all configured noise sources.

        Parameters
        ----------
        n_cycles : int, optional
            Number of modulation cycles to plot. Defaults to 3.
        n_points_per_cycle : int, optional
            Number of points for rendering each cycle. Defaults to 1000.
        noise_seed : int, optional
            A seed for the random number generator to ensure reproducible
            noise. Defaults to 1.

        Returns
        -------
        tuple[matplotlib.axes.Axes, matplotlib.axes.Axes], optional
            The matplotlib axes for the two subplots, or None if simulation
            cannot be run (e.g., fm=0).
        """
        if self.fm == 0:
            logging.warning(
                "Modulation frequency (fm) is zero. Cannot plot a full signal."
            )
            return None, None

        # --- 1. Setup Time Axis and Sampling Rate for Plotting ---
        duration = n_cycles / self.fm
        n_total_samples_plot = n_cycles * n_points_per_cycle
        time_axis = np.linspace(0, duration, n_total_samples_plot, endpoint=False)
        f_samp_plot = float(n_total_samples_plot / duration)

        # --- 2. Create a Minimal IfoConfig for Simulation ---
        dummy_ifo_label = f"{self.label}_plot_ifo"
        dummy_ifo_config = IfoConfig(label=dummy_ifo_label)
        target_m_for_plot = 1e-3
        dummy_delta_l = target_m_for_plot * sc.c / (2 * np.pi * self.df)
        dummy_ifo_config.ref_arml = 0.1
        dummy_ifo_config.meas_arml = dummy_ifo_config.ref_arml + dummy_delta_l

        # --- 3. Create a Dummy SimConfig ---
        dummy_dfmi_channel = SimConfig(
            label=f"{self.label}_signal_plot_channel",
            laser_config=self,  # Use 'self' (this LaserConfig instance)
            ifo_config=dummy_ifo_config,
            f_samp=f_samp_plot,  # Use the calculated sampling rate for plotting
        )

        # --- 4. Generate Noise Arrays and Run Full Simulation Physics ---
        sg = SignalGenerator()
        noise_arrays = sg._generate_noise_arrays(
            dummy_dfmi_channel, n_total_samples_plot, trial_num=noise_seed
        )
        
        witness_freq, witness_phase, _, _ = sg._run_physics_simulation(
            dummy_dfmi_channel,
            time_axis,
            noise_arrays,
            is_dynamic=False,
            timeshift_order=31,
        )
        witness_freq = witness_freq / np.max(witness_freq)  # Normalization

        # --- 5. Create the Plot ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot 1: Voltage signal
        ax1.plot(
            time_axis * 1000, witness_freq, label="Simulated Voltage Signal $v(t)$"
        )
        ax1.set_title(f"Simulated DFMI Signal from Laser '{self.label}' with Noise")
        ax1.set_ylabel("Voltage (a.u.)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # Plot 2: Phase signal
        ax2.plot(
            time_axis * 1000,
            witness_phase,
            label="Total Interferometric Phase $\\Phi_{\\text{tot}}(t)$",
        )
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Phase (rad)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        fig.align_ylabels()

        return ax1, ax2


class SimConfig:
    """Configuration for a complete, single simulation channel.

    This class composes a `LaserConfig` and an `IfoConfig` to describe a
    full DFMI channel. This composite structure allows for clear and
    physically intuitive definitions of complex experiments.

    The key derived parameter, modulation depth `m`, is provided as a
    read-only property calculated from the physical parameters.

    Attributes
    ----------
    label : str
        An identifier for this simulation channel (e.g., 'main_channel').
    laser : LaserConfig
        The laser configuration object for this channel.
    ifo : IfoConfig
        The interferometer configuration object for this channel.
    use_exact_physics : bool
        If True, use the high-fidelity time-shifting physics model. If False,
        use a simplified model. Defaults to True.
    f_samp : float
        The sampling frequency for this channel's data in Hz.
    N : int
        The number of samples in the generated time-series. Set after
        simulation.
    simtime : float, optional
        The wall-clock time taken for the simulation. Set after simulation.
    fit_n : int
        The default number of modulation cycles to average in a fit.
    f_fit : float
        The default fit data rate, calculated as `fm / fit_n`.
    """

    def __init__(self, label, laser_config, ifo_config, f_samp=200e3):
        self.label = label
        self.laser = laser_config  # LaseConfig object
        self.ifo = ifo_config  # IfoConfig object
        self.use_exact_physics = True

        # --- Simulation-specific Parameters ---
        self.f_samp = float(f_samp)  # Sampling frequency (Hz)
        self.N = 0  # Number of samples, set after simulation
        self.simtime = None  # Simulation time, set after simulation
        self.fit_n = 20  # Default number of cycles to average in a fit
        self.f_fit = self.laser.fm / self.fit_n

    def __post_init__(self):
        """Validates the configuration after initialization."""
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string.")
        if not isinstance(self.laser, LaserConfig):
            raise TypeError("laser must be an instance of LaserConfig.")
        if not isinstance(self.ifo, IfoConfig):
            raise TypeError("ifo must be an instance of IfoConfig.")
        if not (math.isfinite(self.f_samp) and self.f_samp > 0.0):
            raise ValueError("f_samp must be a finite, positive float (Hz).")
        if not isinstance(self.fit_n, int) or self.fit_n <= 0:
            raise ValueError("fit_n must be a positive integer.")
        if not (math.isfinite(self.laser.fm) and self.laser.fm > 0.0):
            raise ValueError("laser.fm must be a finite, positive float (Hz).")

        self.f_fit = self.laser.fm / self.fit_n
        if not math.isfinite(self.f_fit) or self.f_fit <= 0.0:
            raise ValueError("f_fit must be finite and positive (Hz).")

    @property
    def m(self) -> float:
        """The effective modulation index (read-only), in radians.

        This value is derived on-the-fly from the physical parameters of the
        associated `LaserConfig` and `IfoConfig` objects. It is not settable
        directly.
        """
        delta_l = np.abs(self.ifo.meas_arml - self.ifo.ref_arml)
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
  Wavelength:          {self.laser.wavelength * 1e6:.3f} um
  Modulation Freq:     {self.laser.fm} Hz
  Modulation Amp (df): {self.laser.df / 1e9:.3f} GHz
  Signal Amplitude:    {self.laser.amp:.2f}

--- Interferometer ('{self.ifo.label}') ---
  Reference Arm:       {self.ifo.ref_arml:.4f} m
  Measurement Arm:     {self.ifo.meas_arml:.4f} m
  OPD (delta_l):       {(self.ifo.meas_arml - self.ifo.ref_arml) * 100:.2f} cm
  Dynamic Motion Amp:  {self.ifo.arml_mod_amp * 1e9:.2f} nm
  Visibility:          {self.ifo.visibility:.2f}

--- Derived & Simulation Parameters ---
  Modulation Depth (m):  {self.m:.4f} rad
  Sampling Freq:         {self.f_samp / 1e3:.1f} kHz
  Downsampling Factor:   {self.f_samp / self.f_fit}
  Fit Cycles (fit_n):    {self.fit_n}
  Output Data Rate (Hz): {self.f_fit}
  Simtime / N:           {f"{self.simtime:.3f}" if self.simtime is not None else "N/A"}{"" if self.simtime is None else " s"} / {self.N if self.N > 0 else "N/A"}
  ============================================================
"""
        logging.info(info_str)

    def plot_harmonics(
        self,
        N: int = 10,
        figsize: Tuple[float, float] = (3.5, 3.5),
        dpi: int = 150,
        ax: Optional[matplotlib.axes.Axes] = None,
        ylim: Optional[float] = None,
    ) -> matplotlib.axes.Axes:
        """Plots the complex harmonic amplitudes of the ideal signal.

        This visualization shows the magnitude and phase of the first N
        harmonics on a polar plot, useful for understanding the
        signal structure in the frequency domain.

        Parameters
        ----------
        N : int, optional
            The number of harmonics to plot. Defaults to 10.
        figsize : tuple, optional
            The figure size if a new plot is created.
        dpi : int, optional
            The figure resolution.
        ax : matplotlib.axes.Axes, optional
            An existing polar Axes object to plot on.
        ylim : float, optional
            The maximum radial limit for the plot.

        Returns
        -------
        matplotlib.axes.Axes
            The polar Axes object containing the plot.
        """
        m = self.m
        phi = self.ifo.phi
        psi = self.laser.psi
        C = self.laser.amp * self.ifo.visibility
        return plot_signal_harmonics(
            phi, psi, m, C, N=N, figsize=figsize, dpi=dpi, ax=ax, ylim=ylim
        )

    def plot_harmonics_vs_phi(
        self, N: int = 5, phi_range: np.ndarray = np.linspace(0, 2 * np.pi, 200)
    ) -> matplotlib.axes.Axes:
        """Plots harmonic amplitudes as a function of interferometric phase.

        This method shows how the real (I_n) and imaginary (Q_n) parts of
        each harmonic's complex amplitude vary as the interferometric phase
        `phi` is swept from 0 to 2*pi.

        Parameters
        ----------
        N : int, optional
            The number of harmonics to plot. Defaults to 5.
        phi_range : np.ndarray, optional
            The range of phi values to sweep over.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the plot.
        """
        m = self.m
        psi = self.laser.psi
        C = self.laser.amp * self.ifo.visibility

        return plot_signal_harmonics_vs_phi(psi, m, C, N=N, phi_range=phi_range)


class SignalGenerator:
    """A dedicated physics engine for generating DFMI time-series data.

    This class encapsulates all the logic for creating realistic DFMI signals,
    including prescribed noise, high-fidelity time-of-flight effects,
    and arbitrary modulation waveforms. It is the primary workhorse for the
    `DeepFrame.simulate()` method.

    The generator supports two main simulation modes:
    1.  **'asd'**: A high-fidelity mode using detailed physical noise models
        based on user-defined Amplitude Spectral Densities (ASDs). This mode
        is suitable for detailed studies of instrument performance.
    2.  **'snr'**: A simplified mode that generates a perfect, noiseless signal
        and adds a specified amount of white Gaussian noise to achieve a
        target Signal-to-Noise Ratio (SNR). This mode is useful for rapid
        algorithm testing.

    Its primary responsibility is to take `SimConfig` objects and produce
    fully populated `RawData` objects containing the simulated time-series
    and all associated ground-truth data.
    """

    def generate(
        self,
        main_config: SimConfig,
        n_seconds: float,
        mode: str = "asd",
        trial_num: int = 0,
        witness_config: Optional[SimConfig] = None,
        snr_db: Optional[float] = None,
        external_noise: Optional[Dict[str, np.ndarray]] = None,
        timeshift_order: int = 31,
        verbose: bool = False,
    ) -> Dict[str, RawData]:
        """Generates one or more linked DFMI signals.

        This is the main public entry point for the physics engine. It routes
        the simulation request to the appropriate internal method based on the
        specified `mode`.

        Parameters
        ----------
        main_config : SimConfig
            The configuration for the primary measurement channel.
        n_seconds : float
            The duration of the time series to generate in seconds.
        mode : {'asd', 'snr'}, optional
            The simulation mode to use. Defaults to 'asd'.
        trial_num : int, optional
            A seed for the random number generators to ensure reproducible
            noise for each trial. Defaults to 0.
        witness_config : SimConfig, optional
            Configuration for a secondary witness channel. If provided, it
            is simulated sharing common noise sources with the main channel.
        snr_db : float, optional
            The target Signal-to-Noise Ratio in dB. Required if `mode='snr'`.
        external_noise : dict, optional
            A dictionary of pre-computed noise time-series. If provided,
            internal noise generation is skipped and these arrays are used.
            Keys must match noise sources (e.g., 'laser_frequency').
        timeshift_order : int, optional
            The order of the Lagrange interpolation filter used for time-
            shifting. Must be an odd integer. Defaults to 31.
        verbose : bool, optional
            If True, enables progress bars and detailed logging.

        Returns
        -------
        dict[str, RawData]
            A dictionary of `RawData` instances, keyed by channel label
            (e.g., 'main', 'witness'). Returns an empty dictionary on failure.
        """
        if (
            not isinstance(main_config, SimConfig)
            or not isinstance(main_config.laser, LaserConfig)
            or not isinstance(main_config.ifo, IfoConfig)
        ):
            raise TypeError(
                "main_config and its laser/ifo attributes must be valid Config objects."
            )

        if witness_config:
            if (
                not isinstance(witness_config, SimConfig)
                or not isinstance(witness_config.laser, LaserConfig)
                or not isinstance(witness_config.ifo, IfoConfig)
            ):
                raise TypeError(
                    "witness_config and its laser/ifo attributes must be valid Config objects."
                )

        if mode == "asd":
            return self._generate_with_asd(
                main_config=main_config,
                n_seconds=n_seconds,
                trial_num=trial_num,
                witness_config=witness_config,
                external_noise=external_noise,
                timeshift_order=timeshift_order,
                verbose=verbose,
            )
        elif mode == "snr":
            if snr_db is None:
                raise ValueError("SNR mode requires a value for 'snr_db'.")
                return {}
            # SNR mode does not support witness channels or external noise for simplicity.
            return self._generate_with_snr(
                main_config=main_config,
                n_seconds=n_seconds,
                trial_num=trial_num,
                snr_db=snr_db,
            )
        else:
            raise ValueError(f"Simulation mode must be 'asd' or 'snr', got: '{mode}'")

    def _generate_with_asd(
        self,
        main_config: SimConfig,
        n_seconds: float,
        trial_num: int = 42,
        witness_config: Optional[SimConfig] = None,
        external_noise: Optional[Dict[str, np.ndarray]] = None,
        timeshift_order: int = 31,
        verbose: bool = False,
    ) -> Dict[str, RawData]:
        """Generates DFMI signals using prescribed noise ASDs.

        This method implements the high-fidelity simulation by modeling noise
        sources from their spectral densities and using high-order interpolation
        for time delays. It employs a "pad-and-crop" strategy to mitigate
        boundary effects from FIR filtering, ensuring the entire output signal
        is valid.

        Parameters are inherited from the public `generate` method.

        Returns
        -------
        dict[str, RawData]
            A dictionary of generated `RawData` objects.
        """
        total_pbar_steps = 7  # Adjust based on how many checkpoints you'll use.
        if verbose:
            pbar = tqdm(total=total_pbar_steps, desc="Running physics simulation...")

        # --- 1. Validate Input and System Parameters ---
        fm = main_config.laser.fm
        f_samp = main_config.f_samp

        # The number of samples per modulation cycle is critical for harmonic analysis.
        samples_per_cycle_float = f_samp / fm
        if not np.isclose(samples_per_cycle_float, round(samples_per_cycle_float)):
            logging.warning(
                f"Non-integer number of samples per cycle ({samples_per_cycle_float:.4f}). "
                "Harmonic orthogonality of the output signal is not guaranteed. "
                "Ensure that f_samp is an integer multiple of fm for best results."
            )
        samples_per_cycle = int(round(samples_per_cycle_float))

        # Determine the number of full modulation cycles requested by the user.
        requested_cycles = int(round(n_seconds * fm))
        if requested_cycles <= 0:
            raise ValueError(
                "The requested simulation time is too short."
                "You must request at least one modulation cycle (1/laser.fm)"
            )

        # --- 2. Calculate Required Padding in Terms of Cycles ---
        # The padding must be sufficient to cover both the maximum physical time-of-flight
        # delay and the half-width of the FIR filter kernel used for time-shifting.

        # A. Find the maximum arm length across all specified channels.
        max_armlength = max(main_config.ifo.ref_arml, main_config.ifo.meas_arml)
        if witness_config:
            max_armlength = max(
                max_armlength, witness_config.ifo.ref_arml, witness_config.ifo.meas_arml
            )

        if verbose:
            pbar.update(1)

        # B. Calculate the maximum physical delay in seconds.
        max_physical_delay_s = max_armlength / sc.c

        # C. Calculate the time corresponding to the FIR filter's half-width.
        filter_half_width_s = ((timeshift_order + 1) // 2) / f_samp

        # D. Sum these times and add a safety margin (e.g., 10%).
        pad_time_s = (max_physical_delay_s + filter_half_width_s) * 1.1

        # E. Convert the required padding time into an integer number of modulation
        #    cycles, rounding up to ensure we have *at least* enough padding.
        pad_cycles = int(np.ceil(pad_time_s * fm))

        logging.debug(
            f"Requested {requested_cycles} cycles. Padding with {pad_cycles} "
            f"cycles on each side to mitigate boundary effects."
        )

        # --- 3. Generate Padded Time Axis and Noise ---
        total_cycles = pad_cycles + requested_cycles + pad_cycles
        num_samples_total = total_cycles * samples_per_cycle
        time_axis_padded = np.arange(num_samples_total) / f_samp
        if verbose:
            pbar.update(1)

        if external_noise:
            logging.debug("Using externally provided noise arrays.")
            noise_keys = ["laser_frequency", "amplitude", "df", "armlength"]
            noise = {key: external_noise.get(key, 0.0) for key in noise_keys}
        else:
            logging.debug("Generating noise internally based on ASDs.")
            noise = self._generate_noise_arrays(
                main_config, num_samples_total, witness_config, trial_num
            )
        if verbose:
            pbar.update(1)

        # --- 4. Run Simulation on Padded Data ---
        main_outputs = self._run_physics_simulation(
            main_config,
            time_axis_padded,
            noise,
            is_dynamic=True,
            timeshift_order=timeshift_order,
        )
        dfmi_signal_padded, dfmi_phase_padded, ground_truth_padded, noisy_ground_truth_padded = main_outputs
        if verbose:
            pbar.update(1)

        # --- 5. Crop Final Outputs to Exact Requested Cycles ---
        num_pad_samples = pad_cycles * samples_per_cycle
        num_final_samples = requested_cycles * samples_per_cycle

        # Define the slice that extracts the central, valid region of the simulation.
        start_index = num_pad_samples
        end_index = start_index + num_final_samples
        valid_slice = slice(start_index, end_index)

        # Crop all main channel outputs.
        dfmi_signal = dfmi_signal_padded[valid_slice]
        dfmi_phase = dfmi_phase_padded[valid_slice]
        simulated_phase_ground_truth = ground_truth_padded[valid_slice]
        noisy_simulated_phase_ground_truth = noisy_ground_truth_padded[valid_slice]

        # Final sanity check on the output length.
        assert len(dfmi_signal) == num_final_samples, "Cropped signal length mismatch."
        if verbose:
            pbar.update(1)

        # --- 6. Package Main Channel Results ---
        raw_main = RawData(data=pd.DataFrame(dfmi_signal, columns=["ch0"]))
        raw_main.label = main_config.label
        raw_main.f_samp = main_config.f_samp
        raw_main.fm = main_config.laser.fm
        raw_main.sim = main_config
        main_config.N = len(dfmi_signal)  # Set final sample count on the config object

        # Store all ground truth and noise signals (also cropped).
        raw_main.phi = dfmi_phase
        raw_main.phi_sim = simulated_phase_ground_truth
        raw_main.phi_sim_noisy = noisy_simulated_phase_ground_truth

        def safe_slice(value, slc):
            return value[slc] if isinstance(value, np.ndarray) else value

        # Apply the safe slicing
        raw_main.f_noise = safe_slice(noise.get("laser_frequency", 0.0), valid_slice)
        raw_main.a_noise = safe_slice(noise.get("laser_rin", 0.0), valid_slice)
        raw_main.l_noise = safe_slice(noise.get("armlength", 0.0), valid_slice)
        raw_main.s_noise = safe_slice(noise.get("electronic", 0.0), valid_slice)
        raw_main.df_noise = safe_slice(noise.get("df", 0.0), valid_slice)

        output_channels = {"main": raw_main}
        if verbose:
            pbar.update(1)

        # --- 7. Handle Witness Channel (if present) ---
        if witness_config is not None:
            # Witness uses the same common noise but its own (static) physics.
            witness_outputs = self._run_physics_simulation(
                witness_config,
                time_axis_padded,
                noise,
                is_dynamic=True,
                is_witness=True,
                timeshift_order=timeshift_order,
            )
            witness_signal_padded, witness_phase_padded, witness_gt_padded, witness_gt_padded_noisy = (
                witness_outputs
            )

            # Crop the witness outputs using the same slice.
            witness_signal = witness_signal_padded[valid_slice]

            # Package the witness channel results.
            raw_witness = RawData(data=pd.DataFrame(witness_signal, columns=["ch0"]))
            raw_witness.label = witness_config.label
            raw_witness.f_samp = witness_config.f_samp
            raw_witness.fm = witness_config.laser.fm
            raw_witness.sim = witness_config
            raw_witness.phi = witness_phase_padded[valid_slice]
            raw_witness.phi_sim = witness_gt_padded[valid_slice]
            raw_witness.phi_sim_noisy = witness_gt_padded_noisy[valid_slice]
            raw_witness.f_noise = safe_slice(
                noise.get("laser_frequency", 0.0), valid_slice
            )
            raw_witness.a_noise = safe_slice(
                noise.get("laser_rin", 0.0), valid_slice
            )
            raw_witness.df_noise = safe_slice(
                noise.get("df", 0.0), valid_slice
            )
            raw_witness.s_noise = safe_slice(
                noise.get("witness_electronic", 0.0), valid_slice
            )
            raw_witness.l_noise = safe_slice(
                noise.get("witness_electronic", 0.0), valid_slice
            )
            output_channels["witness"] = raw_witness

        if verbose:
            pbar.update(1)
            pbar.close()

        return output_channels

    def _generate_with_snr(
        self, main_config: SimConfig, n_seconds: float, trial_num: int, snr_db: float
    ) -> Dict[str, RawData]:
        """Generates a signal with a specific SNR.

        This internal method creates a perfect, noiseless DFMI signal and then
        adds white Gaussian noise to its AC component to achieve a specified
        signal-to-noise ratio.

        Parameters are inherited from the public `generate` method.

        Returns
        -------
        dict[str, RawData]
            A dictionary containing the single generated `RawData` object.
        """
        num_samples = int(n_seconds * main_config.f_samp)
        time_axis = np.arange(num_samples) / main_config.f_samp
        main_config.N = len(time_axis)

        y_clean = self._generate_ideal_signal(main_config, time_axis)
        y_noisy = self._add_white_noise(y_clean, snr_db, trial_num)

        raw_obj = RawData(data=pd.DataFrame(y_noisy, columns=["ch0"]))
        raw_obj.label = main_config.label
        raw_obj.f_samp = main_config.f_samp
        raw_obj.fm = main_config.laser.fm
        raw_obj.t0 = 0
        raw_obj.sim = main_config

        return {"main": raw_obj}

    def _generate_ideal_signal(
        self, sim_config: SimConfig, time_axis: np.ndarray
    ) -> np.ndarray:
        """Generates a perfect, noiseless DFMI signal using the simplified model.

        This helper function is used by the `snr` mode. It calculates the
        signal based on the simplified analytical model:
        v(t) = A * (1 + C * cos(phi + m * cos(omega*t + psi)))

        Parameters
        ----------
        sim_config : SimConfig
            The simulation configuration.
        time_axis : np.ndarray
            The time samples for the simulation.

        Returns
        -------
        np.ndarray
            The ideal, noiseless voltage time-series.
        """
        laser = sim_config.laser
        ifo = sim_config.ifo

        A = laser.amp
        C = ifo.visibility
        omega_mod = 2 * np.pi * laser.fm

        phitot = sim_config.m * np.cos(omega_mod * time_axis + laser.psi)
        return A * (1 + C * np.cos(ifo.phi + phitot))

    def _add_white_noise(
        self, clean_signal: np.ndarray, snr_db: float, trial_num: int
    ) -> np.ndarray:
        """Adds white Gaussian noise to a signal to achieve a target SNR.

        Parameters
        ----------
        clean_signal : np.ndarray
            The input noiseless signal.
        snr_db : float
            The target Signal-to-Noise Ratio in dB.
        trial_num : int
            Seed for the random number generator.

        Returns
        -------
        np.ndarray
            The noisy signal.
        """
        signal_ac = clean_signal - np.mean(clean_signal)
        signal_power = np.mean(signal_ac**2)
        snr_linear_power = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear_power
        noise_std_dev = np.sqrt(noise_power)

        rng = np.random.RandomState(seed=trial_num)
        noise = rng.randn(len(clean_signal)) * noise_std_dev
        return clean_signal + noise

    def _generate_noise_arrays(
        self,
        sim_config: SimConfig, 
        n_samples: int, 
        witness_config: Optional[SimConfig] = None,
        trial_num: int = 0,
        force_start_at_zero: bool = False
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Generates colored random noise arrays for a simulation.

        This function serves as a high-level factory for creating noise time-series
        based on physical specifications (Amplitude Spectral Density and alpha
        exponent). It correctly handles the distinct cases of white and colored
        noise, using the appropriate generator from the `noise` module and applying
        the correct scaling factor to match the target ASD.

        Methodology:
        - White Noise (alpha = 0): The time series is generated from a Gaussian
        distribution whose standard deviation (RMS) is calculated to match the
        total power implied by the constant ASD over the simulation bandwidth
        (up to the Nyquist frequency).
        RMS = ASD * sqrt(f_sample / 2).

        - Colored Noise (0.01 <= alpha <= 2): The `alpha_noise` generator is used,
        which produces a basis signal whose two-sided PSD is 1.0 at 1 Hz.
        To scale this to the target one-sided ASD, a scaling factor is derived:
            - One-sided PSD = 2 * Two-sided PSD (for f>0).
            - The basis signal has a one-sided PSD of 2.0 at 1 Hz.
            - The basis signal has a one-sided ASD of sqrt(2.0) at 1 Hz.
            - The required scaling factor is therefore: `target_ASD / sqrt(2.0)`.

        Parameters
        ----------
        sim_config : SimConfig
            Configuration object containing all noise parameters for the
            simulation, including sampling frequency and noise specs for each
            source.
        n_samples : int
            The number of data samples to generate for each noise array.
        trial_num : int, optional
            A seed offset for the random number generators, ensuring that each
            trial in a Monte Carlo simulation is unique but reproducible.
            Defaults to 0.
        force_start_at_zero : bool, optional
            If True, subtracts the first sample from the entire generated series.
            Useful for some experiment.s

        Returns
        -------
        dict[str, Union[float, np.ndarray]]
            A dictionary where keys are noise source names (e.g.,
            'laser_frequency') and values are the corresponding physically-scaled
            noise time-series arrays. If a source's ASD is zero, the value is 0.0.

        Raises
        ------
        ValueError
            If an ASD is negative or an alpha value is outside the supported
            range Union[0, [0.01, 2]].
        """
        num_samples = int(n_samples)
        if num_samples <= 0:
            raise ValueError(f"n_samples must be > 0, got {num_samples}")

        fs = sim_config.f_samp
        
        # Frequency range is needed for generator initialization
        f_min_res = fs / num_samples  # Lowest possible frequency (resolution)
        f_max_nyq = fs / 2.0          # Nyquist frequency

        noise_sources = {
            # Laser noise sources:
            "laser_frequency": {
                "asd": sim_config.laser.f_n, "alpha": sim_config.laser.f_n_alpha
            },
            "laser_rin": {
                "asd": sim_config.laser.r_n, "alpha": sim_config.laser.r_n_alpha
            },
            "df": {
                "asd": sim_config.laser.df_n, "alpha": sim_config.laser.df_n_alpha
            },
            # Interferometer noise sources:
            "armlength": {
                "asd": sim_config.ifo.arml_n, "alpha": sim_config.ifo.arml_n_alpha
            },
            "electronic": {
                "asd": sim_config.ifo.s_n, "alpha": sim_config.ifo.s_n_alpha
            },
        }

        # Witness interferometer uncorrelated noise sources:
        if witness_config:
            noise_sources["witness_armlength"] = {
                "asd": witness_config.ifo.arml_n, "alpha": witness_config.ifo.arml_n_alpha}
            noise_sources["witness_electronic"] = {
                "asd": witness_config.ifo.s_n, "alpha": witness_config.ifo.s_n_alpha}
            
        # Cache for basis noise generators to avoid re-initializing for the same alpha
        basis_noise_generators: Dict[float, alpha_noise] = {}
        final_noises: Dict[str, Union[float, np.ndarray]] = {}
        
        # Use a single seed counter to ensure all RNGs are unique but deterministic
        # per trial. The large multiplier prevents overlap between trials.
        seed_counter = 1 + trial_num * (len(noise_sources) + 1)

        for name, params in noise_sources.items():
            asd_at_1hz = params.get("asd", 0.0)
            alpha = params.get("alpha", 0.0)

            if asd_at_1hz < 0:
                raise ValueError(f"ASD for '{name}' cannot be negative, got {asd_at_1hz}")

            if asd_at_1hz == 0.0:
                final_noises[name] = 0.0
                continue

            if alpha == 0.0:
                # --- Case 1: White Noise (alpha = 0) ---
                # Total variance for a real signal with a constant one-sided PSD
                # of ASD^2 is Var = integral(ASD^2, 0, fs/2) = ASD^2 * (fs/2).
                # The standard deviation (RMS) is therefore ASD * sqrt(fs/2).
                sigma = asd_at_1hz * np.sqrt(f_max_nyq)
                
                # We use np.random directly for efficiency.
                rng = np.random.default_rng(seed_counter)
                seed_counter += 1
                noise_series = rng.normal(scale=sigma, size=num_samples)

            elif 0.01 <= alpha <= 2.0:
                # --- Case 2: Colored Noise (0.01 <= alpha <= 2) ---
                
                # 2a. Get a basis noise generator (from cache or create a new one).
                if alpha not in basis_noise_generators:
                    # The alpha_noise class whitens below f_min, so we use the
                    # simulation's frequency resolution as the lower bound to
                    # capture all relevant dynamics.
                    basis_noise_generators[alpha] = alpha_noise(
                        f_sample=fs,
                        f_min=f_min_res,
                        f_max=f_max_nyq,
                        alpha=alpha,
                        init_filter=True, # Settle filter for realistic noise
                        seed=seed_counter,
                    )
                    seed_counter += 1
                
                generator = basis_noise_generators[alpha]
                
                # 2b. Generate the basis noise series.
                # This series has a two-sided PSD of 1.0 at 1 Hz.
                basis_series = generator.get_series(num_samples)
                
                # 2c. Calculate the correct scaling factor and apply it.
                # The one-sided ASD of the basis series at 1 Hz is sqrt(2.0).
                # We scale it to match our target ASD.
                scaling_factor = asd_at_1hz / np.sqrt(2.0)
                noise_series = basis_series * scaling_factor
                
            else:
                raise ValueError(
                    f"Alpha for '{name}' is invalid. Must be 0.0 or in [0.01, 2], got {alpha}."
                )

            # Optional high-pass filtering for special cases.
            if force_start_at_zero:
                noise_series = noise_series - noise_series[0]

            final_noises[name] = noise_series

        return final_noises

    def _run_physics_simulation(
        self,
        sim_config: SimConfig,
        time_axis: np.ndarray,
        noise_arrays: Dict[str, np.ndarray],
        is_dynamic: bool,
        is_witness: bool = False,
        timeshift_order: int = 31,
        oversampling_factor: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Core physics calculation for a single channel.

        This internal method implements the physical model for an
        unequal-arm interferometer. It calculates the final interferometric signal
        by modeling the time-of-flight delays of a frequency-modulated laser.
        All noise sources are injected at their physically appropriate locations.

        Parameters
        ----------
        sim_config : SimConfig
            The complete configuration for the channel.
        time_axis : np.ndarray
            The time samples for the simulation.
        noise_arrays : dict
            A dictionary of pre-generated noise time-series arrays.
        is_dynamic : bool
            If True, simulates dynamic arm length changes.
        timeshift_order : int
            The interpolation order for `dsp.timeshift`.
        oversampling_factor : int
            Oversampling used for numerical integration of custom waveforms.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - voltage_signal: The final simulated voltage from the photodetector.
            - dfmi_phase: The total, noisy interferometric phase.
            - simulated_phase_ground_truth: The ideal, noiseless phase signal
              that a perfect instrument should measure.
        """
        # --- 1. Unpack Configuration ---
        laser = sim_config.laser
        ifo = sim_config.ifo
        fs = sim_config.f_samp
        omega_mod = 2 * np.pi * laser.fm
        t_phase = omega_mod * time_axis + laser.psi

        # --- 2. Calculate Dynamic Path and its Direct Phase Effect ---
        if is_dynamic:
            dynamic_path_change = ifo.arml_mod_amp * np.sin(
                2 * np.pi * ifo.arml_mod_f * time_axis + ifo.arml_mod_psi
            )
            if is_witness:
                dynamic_path_change += noise_arrays.get("witness_armlength", 0.0)
            else:
                dynamic_path_change += noise_arrays.get("armlength", 0.0)
        else:
            dynamic_path_change = 0.0
        dynamic_opd_change = dynamic_path_change
        omega_0_clean = 2 * np.pi * sc.c / laser.wavelength
        dynamic_carrier_phase_signal = omega_0_clean * dynamic_opd_change / sc.c

        # --- 3. Calculate the Differential Laser Modulation Term ---
        if sim_config.use_exact_physics:
            # A. Calculate the physical amplitude of the phase modulation's fundamental tone.
            if laser.fm > 0:
                phi_mod_amp_fundamental = laser.df / laser.fm
            else:
                raise ValueError("Cannot run physics model when laser 'df' is zero.")

            # B. HYBRID LOGIC: Choose path based on the specific waveform function.
            # --- Analytical Path 1: Pure Cosine (Ideal DFMI) ---
            if laser.waveform_func in [waveforms.cosine, np.cos]:
                # The integral of Δf*cos(ωt) is (Δf/f_m)*sin(ωt).
                phi_mod_unscaled_physical = phi_mod_amp_fundamental * np.sin(t_phase)

            # --- Analytical Path 2: Second Harmonic Distortion ---
            elif laser.waveform_func is waveforms.shd:
                # This logic is taken directly from the original, high-accuracy engine.
                kwargs = laser.waveform_kwargs
                dist_amp = kwargs.get("distortion_amp", 0.0)
                dist_phase = kwargs.get("distortion_phase", 0.0)

                # The integral of A*cos(2ωt+p) is (A/2)*sin(2ωt+p).
                phi_mod_amp_distortion = (phi_mod_amp_fundamental * dist_amp) / 2.0

                phi_mod_unscaled_physical = phi_mod_amp_fundamental * np.sin(
                    t_phase
                ) + phi_mod_amp_distortion * np.sin(2 * t_phase + dist_phase)

            # --- Numerical Fallback Path: For all other custom waveforms ---
            else:
                # Create oversampled axes
                num_samples_os = len(time_axis) * oversampling_factor
                time_axis_os = np.linspace(
                    time_axis[0], time_axis[-1], num=num_samples_os, endpoint=False
                )
                t_phase_os = omega_mod * time_axis_os + laser.psi

                # Normalize frequency shape
                fm_shape_os = laser.waveform_func(t_phase_os, **laser.waveform_kwargs)
                fm_shape_ac_os = fm_shape_os - np.mean(fm_shape_os)
                fund_amp_in_shape = np.mean(fm_shape_ac_os * np.cos(t_phase_os)) * 2
                if abs(fund_amp_in_shape) > 1e-9:
                    fm_shape_normalized_os = fm_shape_ac_os / fund_amp_in_shape
                else:
                    fm_shape_normalized_os = fm_shape_ac_os

                # Use robust trapezoid integrator; accuracy is ensured by high oversampling.
                phi_shape_unscaled_os = cumulative_trapezoid(
                    fm_shape_normalized_os, x=t_phase_os, initial=0
                )

                # Detrend to enforce periodicity
                total_phase_span_os = t_phase_os[-1] - t_phase_os[0]
                if total_phase_span_os > 0:
                    endpoint_error = phi_shape_unscaled_os[-1]
                    drift_slope = endpoint_error / total_phase_span_os
                    correction_ramp = drift_slope * (t_phase_os - t_phase_os[0])
                    phi_shape_periodic_os = phi_shape_unscaled_os - correction_ramp
                else:
                    phi_shape_periodic_os = phi_shape_unscaled_os

                phi_mod_shape_ac_os = phi_shape_periodic_os - np.mean(
                    phi_shape_periodic_os
                )

                # Use anti-aliasing downsampler for maximum accuracy and integrity.
                phi_mod_shape = decimate(
                    phi_mod_shape_ac_os,
                    oversampling_factor,
                    ftype="fir",
                    zero_phase=True,
                )

                # Safely handle any potential length mismatch by truncating.
                final_len = len(time_axis)
                if len(phi_mod_shape) != final_len:
                    phi_mod_shape = phi_mod_shape[:final_len]

                # Apply the physical amplitude
                phi_mod_unscaled_physical = phi_mod_amp_fundamental * phi_mod_shape

            # C. Inject modulation amplitude (df) noise. This applies to all paths.
            if laser.df != 0:
                df_noise_factor = 1 + (noise_arrays.get("df", 0.0) / laser.df)
                phi_mod_waveform = phi_mod_unscaled_physical * df_noise_factor
            else:
                phi_mod_waveform = phi_mod_unscaled_physical

            # D. Calculate time delays and apply time shifts.
            tau_r_roundtrip = ifo.ref_arml / sc.c
            tau_m_roundtrip_static = ifo.meas_arml / sc.c
            tau_dl_dynamic_roundtrip = dynamic_opd_change / sc.c

            shifts_r_samples = -(tau_r_roundtrip * fs)
            shifts_m_samples = -(
                (tau_m_roundtrip_static + tau_dl_dynamic_roundtrip) * fs
            )

            phi_mod_ref = timeshift(
                phi_mod_waveform, shifts_r_samples, order=timeshift_order
            )
            phi_mod_meas = timeshift(
                phi_mod_waveform, shifts_m_samples, order=timeshift_order
            )
            delta_phi_mod = phi_mod_ref - phi_mod_meas

        else:  # Approximated Model
            m_noisy = sim_config.m * (
                1 + (noise_arrays.get("df", 0.0) / laser.df if laser.df != 0 else 0)
            )
            delta_phi_mod = m_noisy * np.cos(t_phase)

        # --- 4. Construct the Final Signal with ALL Phase Components ---
        f_noise = noise_arrays.get("laser_frequency", 0.0)

        # High-fidelity laser frequency noise injection for the exact model
        if isinstance(f_noise, np.ndarray) and sim_config.use_exact_physics:
            # Re-calculate shifts in case this is the first time they are computed
            if "shifts_r_samples" not in locals():
                tau_r_roundtrip = ifo.ref_arml / sc.c
                tau_m_roundtrip_static = ifo.meas_arml / sc.c
                tau_dl_dynamic_roundtrip = dynamic_opd_change / sc.c
                shifts_r_samples = -(tau_r_roundtrip * fs)
                shifts_m_samples = -(
                    (tau_m_roundtrip_static + tau_dl_dynamic_roundtrip) * fs
                )

            f_noise_ref = timeshift(f_noise, shifts_r_samples, order=timeshift_order)
            f_noise_meas = timeshift(f_noise, shifts_m_samples, order=timeshift_order)
            differential_f_noise = f_noise_ref - f_noise_meas
            phase_noise_from_laser = (
                2
                * np.pi
                * cumulative_trapezoid(differential_f_noise, dx=1 / fs, initial=0)
            )
        else:
            # Use approximation for the non-exact model or if no noise is present
            static_opd_roundtrip = ifo.meas_arml - ifo.ref_arml
            phase_noise_from_laser = 2 * np.pi * f_noise * static_opd_roundtrip / sc.c

        # Sum all phase components:
        # 1. Static phase offset + direct phase from arm motion
        simulated_phase_ground_truth = ifo.phi + dynamic_carrier_phase_signal
        # 2. + Residual differential laser phase noise
        simulated_noisy_phase_ground_truth = simulated_phase_ground_truth + phase_noise_from_laser
        # 3. + Differential phase from laser's own FM
        dfmi_phase = simulated_noisy_phase_ground_truth + delta_phi_mod

        # Assemble final signal:
        amplitude_effective = laser.amp*(1 + noise_arrays.get("laser_rin", 0.0))

        voltage_signal = amplitude_effective * (
            1 + ifo.visibility * np.cos(dfmi_phase)
        ) 
        if is_witness:
            voltage_signal += noise_arrays.get("witness_electronic", 0.0)
        else:
            voltage_signal += noise_arrays.get("electronic", 0.0)

        return voltage_signal, dfmi_phase, simulated_phase_ground_truth, simulated_noisy_phase_ground_truth
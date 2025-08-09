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
from deepfmkit.data import RawData, FitData, open_txt_maybe_gzip
from deepfmkit.physics import IfoConfig, LaserConfig, SimConfig, SignalGenerator
from deepfmkit.fitters import StandardNLSFitter, EKFFitter
from deepfmkit.dsp import vectorized_downsample
from deepfmkit.plotting import plot_fit, plot_fit_comparison, plot_fit_difference

import numpy as np
import scipy.constants as sc
import pandas as pd
import time
from typing import Optional, List, Dict, Any

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class DeepFrame:
    """A framework for loading, processing, analyzing, and plotting DFMI data.

    This class acts as the main controller for the DeepFMKit toolkit. It holds
    the state of an analysis session, including configurations, raw data, and
    fit results, and provides high-level methods to orchestrate the entire
    simulation and analysis workflow.

    Parameters
    ----------
    sim_obj : SimConfig, optional
        A simulation configuration object to load upon initialization.
    raw_file : str, optional
        Path to a raw data file to load upon initialization.
    fit_file : str, optional
        Path to a fit data file to load upon initialization.
    raw_labels : list of str, optional
        Labels to assign to channels from `raw_file`.
    fit_labels : list of str, optional
        Labels to assign to channels from `fit_file`.

    Attributes
    ----------
    lasers : dict[str, LaserConfig]
        Dictionary of laser source configurations.
    ifos : dict[str, IfoConfig]
        Dictionary of interferometer configurations.
    sims : dict[str, SimConfig]
        Dictionary of complete simulation channel configurations.
    raws : dict[str, RawData]
        Dictionary of raw time-series data objects.
    fits : dict[str, FitData]
        Dictionary of fit result objects.
    fits_df : dict[str, pd.DataFrame]
        Dictionary of pandas DataFrames containing raw fit results.
    """

    def __init__(
        self,
        sim_obj: Optional[SimConfig] = None,
        raw_file: Optional[str] = None,
        fit_file: Optional[str] = None,
        raw_labels: Optional[List[str]] = None,
        fit_labels: Optional[List[str]] = None,
    ):
        self.raw_file = raw_file  # Path to raw data file
        self.fit_file = fit_file  # Path to fit data file
        self.lasers: Dict[str, LaserConfig] = {}
        self.ifos: Dict[str, IfoConfig] = {}
        self.sims: Dict[str, SimConfig] = {}
        self.raws: Dict[str, RawData] = {}
        self.fits: Dict[str, FitData] = {}
        self.fits_df: Dict[str, pd.DataFrame] = {}
        self.channr: Optional[int] = None
        self.n: Optional[int] = None
        self.t0: Optional[int] = None
        self.R: Optional[int] = None
        self.fs: Optional[float] = None
        self.f_samp: Optional[float] = None
        self.fm: Optional[float] = None
        self.ndata: int = 10
        self.init_a: float = 1.6
        self.init_m: float = 6.0

        if self.raw_file is not None:
            self.load_raw(labels=raw_labels)

        if self.fit_file is not None:
            self.load_fit(labels=fit_labels)

        if sim_obj is not None:
            self.add_sim(sim_obj)

    def to_txt(self, filepath: str = "./", labels: Optional[List[str]] = None) -> None:
        """Saves fit data to text files in the standard DFMI format.

        For each specified label, this method retrieves the corresponding
        FitData object and calls its `to_txt()` method to write the data
        to a file.

        Parameters
        ----------
        filepath : str, optional
            The directory where the output files will be saved. Defaults to
            the current directory.
        labels : list of str, optional
            A list of fit labels to save. If None, all fits currently stored
            in `self.fits` will be saved. Defaults to None.
        """
        if labels is not None:
            for label in labels:
                filename = filepath + label + ".txt"
                self.fits[label].to_txt(filename)
        else:
            for fit in self.fits:
                filename = filepath + self.fits[fit].label + ".txt"
                self.fits[fit].to_txt(filename)

    def _parse_header(self, file_select: str = "raw") -> None:
        """Parses the header of a raw_data or fit_data file.

        This internal helper reads the first 11 lines of a specified file
        to extract metadata and populate the corresponding attributes of the
        DeepFrame instance (e.g., `f_samp`, `fm`).

        Parameters
        ----------
        file_select : {'raw', 'fit'}, optional
            Specifies whether to parse the header of `self.raw_file` or
            `self.fit_file`. Defaults to 'raw'.

        Notes
        -----
        This function's logic is specific to the header format used in the
        DFMSWPM project and would need revision if the format changes.
        """
        lines = []
        values = []

        file_to_parse = None
        if file_select == "raw":
            file_to_parse = self.raw_file
        elif file_select == "fit":
            file_to_parse = self.fit_file

        if not file_to_parse:
            raise FileNotFoundError(
                f"No {file_select} file has been specified in the DeepFrame object."
            )

        if file_select == "raw":
            with open_txt_maybe_gzip(self.raw_file) as f:
                for _ in range(11):
                    lines.append(f.readline())
        elif file_select == "fit":
            with open_txt_maybe_gzip(self.fit_file) as f:
                for _ in range(11):
                    lines.append(f.readline())

        for v in range(2, 11):
            values.append("".join([c for c in lines[v] if c in "1234567890."]))

        if file_select == "raw":
            self.channr = int(values[0])
            self.t0 = int(values[1])
            self.f_samp = float(values[2])
            self.fm = float(values[3])
            logging.info("Number of channels: {}".format(self.channr))
            logging.info("Starting time: {}".format(self.t0))
            logging.info("Sampling frequency: {}".format(self.f_samp))
            logging.info("Modulation frequency: {}".format(self.fm))
        else:
            self.channr = int(values[0])
            self.t0 = int(values[1])
            self.f_samp = float(values[2])
            self.fm = float(values[3])
            self.n = int(values[4])
            self.R = int(values[5])
            self.fs = float(values[6])
            logging.info("Number of channels: {}".format(self.channr))
            logging.info("Starting time: {}".format(self.t0))
            logging.info("Sampling frequency: {}".format(self.f_samp))
            logging.info("Modulation frequency: {}".format(self.fm))
            logging.info("n: {}".format(self.n))
            logging.info("Downsampling factor: {}".format(self.R))
            logging.info("Fit data rate: {}".format(self.fs))

    def simulate(
        self,
        label: str,
        n_seconds: float,
        mode: str = "asd",
        witness_label: Optional[str] = None,
        snr_db: Optional[float] = None,
        trial_num: int = 0,
        timeshift_order: int = 31,
        verbose: bool = False,
    ) -> None:
        """Orchestrates a physics simulation using the SignalGenerator.

        This method is the main entry point for generating synthetic data. It
        retrieves simulation configurations from `self.sims`, passes them
        to the physics engine, and stores the resulting `RawData` objects
        in `self.raws`.

        Parameters
        ----------
        label : str
            The label of the SimConfig in `self.sims` to use for the main
            channel.
        n_seconds : float
            The duration of the simulation in seconds.
        mode : {'asd', 'snr'}, optional
            The simulation mode. 'asd' uses the user-prescribed noise
            amplitude spectral densities. 'snr' generates a perfect
            signal and adds white noise to achieve a target SNR.
            Defaults to 'asd'.
        witness_label : str, optional
            The label of the SimConfig to use for a witness channel. If
            provided, a second channel is generated which shares common noise
            sources with the main channel. Defaults to None.
        snr_db : float, optional
            The target Signal-to-Noise Ratio in dB. Required if `mode='snr'`.
        trial_num : int, optional
            A number used to seed the random noise generators, ensuring
            reproducible results. Defaults to 0.
        timeshift_order : int, optional
            The order of the Lagrange interpolator for time-shifting.
            Must be an odd integer. Defaults to 31.
        verbose : bool, optional
            If True, enables detailed logging from the simulation engine.
        """
        t0 = time.time()

        # --- 1. Validate configurations ---
        if label not in self.sims:
            raise KeyError(f"Main simulation label '{label}' not found in self.sims.")
        main_config = self.sims[label]

        witness_config = None
        if witness_label:
            if witness_label not in self.sims:
                raise KeyError(
                    f"Witness simulation label '{witness_label}' not found in self.sims."
                )
            witness_config = self.sims[witness_label]
            logging.debug(
                f"Simulating '{label}' with witness '{witness_label}' for {n_seconds:.2f}s..."
            )
        else:
            logging.debug(f"Simulating '{label}' for {n_seconds:.2f}s...")

        # --- 2. Instantiate and run the physics engine ---
        generator = SignalGenerator()
        generated_channels = generator.generate(
            main_config=main_config,
            n_seconds=n_seconds,
            mode=mode,
            trial_num=trial_num,
            witness_config=witness_config,
            snr_db=snr_db,
            timeshift_order=timeshift_order,
            verbose=verbose,
        )

        # --- 3. Store the results in the framework ---
        if not generated_channels:
            logging.warning(
                "Simulation ran but failed to generate data. Check simulation parameters."
            )
            return

        for _, raw_obj in generated_channels.items():
            self.raws[raw_obj.label] = raw_obj

        sim_time = time.time() - t0
        main_config.simtime = sim_time
        logging.debug(f"Simulation finished in {sim_time:.3f} s.")

    def add_sim(self, sim: SimConfig) -> None:
        """Loads a SimConfig object into the framework.

        Parameters
        ----------
        sim : SimConfig
            The simulation configuration object to add to `self.sims`.
        """
        if not isinstance(sim, SimConfig):
            raise TypeError("Argument must be a valid SimConfig object.")
        self.sims[sim.label] = sim

    def new_sim(self, label: Optional[str] = None) -> str:
        """Creates and registers a new, default SimConfig object.

        This is a convenience method for quickly adding a new simulation
        channel to the framework. The user must subsequently populate the
        `laser` and `ifo` attributes of the new `SimConfig` object.

        Parameters
        ----------
        label : str, optional
            A label for the new simulation. If None, a timestamp-based label
            is automatically generated. Defaults to None.

        Returns
        -------
        str
            The label assigned to the newly created SimConfig.
        """
        if label is None:
            from datetime import datetime

            label = datetime.now().strftime("%Y%m%d_%H%M%S")

        sim = SimConfig(label=label, laser_config=LaserConfig(), ifo_config=IfoConfig())
        self.sims[sim.label] = sim
        return label

    def load_raw(
        self, raw_file: Optional[str] = None, labels: Optional[List[str]] = None
    ) -> None:
        """Loads raw data from a text file into RawData objects.

        This method parses a standard DFMI raw data file, which can contain
        one or more channels of time-series data. It creates a `RawData`
        object for each channel and stores them in `self.raws`.

        Parameters
        ----------
        raw_file : str, optional
            Path to the raw data file. If None, uses the path stored in
            `self.raw_file`. Defaults to None.
        labels : list of str, optional
            A list of labels to assign to the loaded channels. The length
            must match the number of channels in the file. If None,
            labels are auto-generated. Defaults to None.
        """
        if raw_file is not None:
            self.raw_file = raw_file

        if self.raw_file is None:
            raise ValueError("Must specify a file to load.")

        self._parse_header(file_select="raw")

        if labels is None:
            labels = [""] * self.channr
            for c in range(self.channr):
                labels[c] = self.raw_file + "_ch" + str(c)
        else:
            assert len(labels) == self.channr

        for c in range(self.channr):
            raw = RawData(
                data=pd.read_csv(
                    self.raw_file,
                    sep=" ",
                    skiprows=13,
                    usecols=[c],
                    names=["ch" + str(c)],
                )
            )
            raw.raw_file = self.raw_file
            raw.label = labels[c]
            raw.t0 = self.t0
            raw.f_samp = self.f_samp
            raw.fm = self.fm
            self.raws[raw.label] = raw

    def load_fit(
        self, fit_file: Optional[str] = None, labels: Optional[List[str]] = None
    ) -> None:
        """Loads fit data from a text file into FitData objects.

        This method parses a standard DFMI fit data file, creating a `FitData`
        object for each channel of results and storing them in `self.fits`.

        Parameters
        ----------
        fit_file : str, optional
            Path to the fit data file. If None, uses the path stored in
            `self.fit_file`. Defaults to None.
        labels : list of str, optional
            A list of labels to assign to the loaded channels. The length
            must match the number of channels in the file. If None,
            labels are auto-generated. Defaults to None.
        """
        if fit_file is not None:
            self.fit_file = fit_file

        if self.fit_file is None:
            raise ValueError("Must specify a file to load.")

        self._parse_header(file_select="fit")

        if labels is None:
            labels = [""] * self.channr
            for c in range(self.channr):
                labels[c] = f"{self.fit_file}_ch{str(c)}"
        else:
            assert len(labels) == self.channr

        data = np.genfromtxt(
            self.fit_file, dtype="double", skip_header=11, invalid_raise=False
        )

        for k in range(self.channr):
            fit = FitData()
            fit.nbuf = len(data[:, 0])
            fit.n = self.n
            fit.t0 = self.t0
            fit.R = self.R
            fit.fs = self.fs
            fit.f_samp = self.f_samp
            fit.fm = self.fm
            fit.ndata = self.ndata
            fit.ssq = data[:, 6 * k + 0]
            fit.amp = data[:, 6 * k + 1]
            fit.m = data[:, 6 * k + 2]
            fit.phi = data[:, 6 * k + 3]
            fit.psi = data[:, 6 * k + 4]
            fit.dc = data[:, 6 * k + 5]
            fit.time = np.arange(0, fit.nbuf / self.fs, 1.0 / self.fs)
            fit.label = labels[k]
            self.fits[labels[k]] = fit

    def _create_fit_object_from_df(
        self,
        fit_label: str,
        source_label: str,
        n: int,
        R: int,
        fs: float,
        nbuf: int,
        ndata: int,
        init_a: float,
        init_m: float,
    ) -> FitData:
        """Creates and registers a FitData object from a results DataFrame.

        This internal helper converts the raw pandas DataFrame produced by a
        fitter into a fully-populated `FitData` object. It handles the
        assignment of all metadata from the source `RawData` and the fit
        configuration, and registers the final object in `self.fits`.

        Parameters
        ----------
        fit_label : str
            The label for the new FitData and the key for `self.fits`.
        source_label : str
            The label of the source RawData in `self.raws`.
        n : int
            Number of modulation cycles per fit buffer.
        R : int
            Buffer size in samples (downsampling factor).
        fs : float
            Fit data rate in Hz.
        nbuf : int
            Total number of buffers in the fit.
        ndata : int
            Number of harmonics used in the fit.
        init_a : float
            Initial amplitude guess used for the fit.
        init_m : float
            Initial modulation depth guess used for the fit.

        Returns
        -------
        FitData
            The newly created and registered fit object.
        """
        df = self.fits_df[fit_label]

        fit = FitData()
        fit.n, fit.R, fit.fs, fit.nbuf, fit.ndata, fit.init_a, fit.init_m = (
            n,
            R,
            fs,
            nbuf,
            ndata,
            init_a,
            init_m,
        )

        fit.t0 = self.raws[source_label].t0
        fit.f_samp = self.raws[source_label].f_samp
        fit.fm = self.raws[source_label].fm

        fit.ssq = df["ssq"].to_numpy()
        fit.amp = df["amp"].to_numpy()
        fit.m = df["m"].to_numpy()
        fit.tau = df["tau"].to_numpy()
        fit.phi = df["phi"].to_numpy()
        fit.psi = df["psi"].to_numpy()
        fit.dc = df["dc"].to_numpy()
        fit.time = np.arange(0, fit.ssq.shape[0] / fit.fs, 1.0 / fit.fs)
        fit.label = fit_label

        self.fits[fit_label] = fit
        return fit

    def _fit_init(self, label: str, n: int) -> tuple[int, float, int]:
        """Calculates key parameters for a fitting run.

        Based on the number of modulation cycles `n` to include in each fit
        buffer, this function calculates the resulting buffer size in samples,
        the fit rate, and the total number of buffers available in a dataset.

        Parameters
        ----------
        label : str
            The label of the source RawData in `self.raws`.
        n : int
            The number of modulation periods (`fm`) to include in each
            analysis buffer.

        Returns
        -------
        tuple[int, float, int]
            A tuple containing (R, fs, nbuf):
            - R: The buffer size in samples.
            - fs: The resulting fit data rate in Hz.
            - nbuf: The total number of full buffers available in the data.

        Raises
        ------
        ValueError
            If the calculated number of buffers is zero.
        """
        R = int(self.raws[label].f_samp / self.raws[label].fm * n)
        fs = self.raws[label].f_samp / R
        nbuf = int(self.raws[label].data.shape[0] / R)
        if nbuf == 0:
            raise ValueError(
                f"Buffer configuration for '{label}' results in 0 buffers. "
                "The data is too short for the requested `n` value."
            )

        return R, fs, nbuf

    def fit(
        self,
        main_label: str,
        method: str = "nls",
        fit_label: Optional[str] = None,
        **kwargs: Any,
    ) -> FitData:
        """Fits raw data using a specified algorithm.

        This unified fitting interface selects a fitter class based on the
        `method` string, instantiates it, and executes the fit. It acts as a
        dispatcher using the Strategy design pattern.

        Parameters
        ----------
        main_label : str
            The label of the `RawData` object in `self.raws` to be fit.
        method : str, optional
            The fitting algorithm to use. Available methods: 'nls', 'ekf'.
            Defaults to 'nls'.
        fit_label : str, optional
            The label for the output `FitData`. If None, a label is
            auto-generated. Defaults to None.
        **kwargs : Any
            Additional keyword arguments passed directly to the selected
            fitter's `fit()` method.

        Returns
        -------
        FitData
            A fit object containing the results, which is also stored in
            `self.fits`. Returns None on failure.
        """
        # --- 1. Select the Fitter Class ---
        fitter_map = {
            "nls": StandardNLSFitter,
            "ekf": EKFFitter,
        }
        if method not in fitter_map:
            raise ValueError(
                f"Unknown fit method: '{method}'. Available: {list(fitter_map.keys())}"
            )

        FitterClass = fitter_map[method]
        logging.debug(
            f"Dispatching to {FitterClass.__name__} for label '{main_label}'."
        )

        # --- 2. Prepare Data and Config ---
        if main_label not in self.raws:
            raise KeyError(
                f"Invalid raw data label: '{main_label}' not found in self.raws."
            )
        main_raw = self.raws[main_label]

        if fit_label is None:
            fit_label = f"{main_label}_{method}"

        # Get 'n' for the fit config, which is common to all fitters
        n_cycles = kwargs.get("n")
        if n_cycles is None:
            sim_obj = self.sims.get(main_raw.sim.label if main_raw.sim else main_label)
            n_cycles = sim_obj.fit_n if sim_obj else 20

        # The base config for all fitters
        fit_config = {"n": n_cycles}

        R, fs, nbuf = self._fit_init(main_label, n_cycles)
        if (
            hasattr(main_raw, "phi_sim")
            and main_raw.phi_sim is not None
            and len(main_raw.phi_sim) > 0
        ):
            main_raw.phi_sim_downsamp = vectorized_downsample(main_raw.phi_sim, R)

        # --- 3. Instantiate and Run the Fitter ---
        fitter_args = {"main_raw": main_raw}

        fitter = FitterClass(fit_config)
        results_df = fitter.fit(**fitter_args, **kwargs)

        # --- 4. Create and Store the Final FitData ---
        if results_df is None or results_df.empty:
            logging.warning(
                f"{FitterClass.__name__} returned no results for '{main_label}'."
            )
            return None

        if method in ["nls", "ekf"]:
            results_df["tau"] = (
                results_df["m"] / (2 * np.pi * main_raw.sim.laser.df)
                if main_raw.sim
                else 0.0
            )

        self.fits_df[fit_label] = results_df

        fit_obj = self._create_fit_object_from_df(
            fit_label,
            main_label,
            n_cycles,
            R,
            fs,
            nbuf,
            fit_config.get("ndata", 0),
            0,
            0,
        )
        self.fits[fit_label] = fit_obj

        return fit_obj

    def create_witness_channel(
        self,
        main_channel_label: str,
        witness_channel_label: str,
        m_witness: Optional[float] = None,
        delta_l_witness: Optional[float] = None,
    ) -> None:
        """Creates a witness channel with optional auto-tuning.

        This helper creates a static witness channel linked to a main channel's
        laser. It sets the witness arm lengths to achieve a target `m_witness`
        or `delta_l_witness`.

        Parameters
        ----------
        main_channel_label : str
            The label of the existing SimConfig to get the shared laser.
        witness_channel_label : str
            The label for the new witness channel to be created.
        m_witness : float, optional
            The target effective modulation index for the witness channel.
        delta_l_witness : float, optional
            The desired absolute optical path difference for the witness,
            in meters.

        Raises
        ------
        KeyError
            If `main_channel_label` is not found.
        ValueError
            If both `m_witness` and `delta_l_witness` are specified.
        """
        # --- 1. Validate inputs and get shared laser ---
        if main_channel_label not in self.sims:
            raise KeyError(
                f"Main channel '{main_channel_label}' not found in framework."
            )
        if delta_l_witness is not None and m_witness is not None:
            raise ValueError(
                "Please specify either delta_l_witness or m_witness, but not both."
            )

        main_channel = self.sims[main_channel_label]
        shared_laser = main_channel.laser

        # --- 2. Determine target witness modulation depth ---
        if m_witness is None and delta_l_witness is None:
            m_target = 0.1
        elif m_witness is not None:
            m_target = m_witness
        else:
            m_target = (2 * np.pi * shared_laser.df * delta_l_witness) / sc.c

        # --- 3. Create and configure witness interferometer ---
        witness_ifo = IfoConfig(label=f"{witness_channel_label}_ifo")
        witness_ifo.arml_mod_amp = 0.0
        witness_ifo.arml_n = 0.0

        if shared_laser.df == 0:
            raise ValueError("Cannot set 'm_witness' when laser 'df' is zero.")
        final_delta_l = (m_target * sc.c) / (2 * np.pi * shared_laser.df)
        witness_ifo.ref_arml = 0.01
        witness_ifo.meas_arml = witness_ifo.ref_arml + final_delta_l

        f0 = sc.c / shared_laser.wavelength
        static_fringe_phase = (2 * np.pi * f0 * final_delta_l) / sc.c
        phi_offset_required = (np.pi / 2.0) + static_fringe_phase
        witness_ifo.phi = phi_offset_required % (2 * np.pi)

        # --- 4. Compose and register the new channel ---
        witness_channel = SimConfig(
            label=witness_channel_label,
            laser_config=shared_laser,
            ifo_config=witness_ifo,
            f_samp=main_channel.f_samp,
        )
        witness_channel.fit_n = main_channel.fit_n
        self.sims[witness_channel_label] = witness_channel

        logging.debug(
            f"Created witness channel '{witness_channel_label}' with final m_witness={witness_channel.m:.3f}."
        )

    def plot(
        self,
        labels: Optional[List[str]] = None,
        which: Optional[List[str]] = None,
        figsize: Optional[tuple] = None,
        dpi: int = 150,
        xrange: Optional[tuple] = None,
        styles: Optional[List[str]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Plots selected fit data for one or more channels.

        This is a high-level wrapper around the `plot_fit` function.

        Parameters
        ----------
        labels : list of str, optional
            A list of fit labels to plot from `self.fits`. If None, plots all
            available fits. Defaults to None.
        which : list of str, optional
            A subset of ['psi', 'phi', 'm', 'amp', 'dc', 'ssq'] to plot.
            If None, all are plotted. Defaults to None.
        figsize : tuple, optional
            The figure size (width, height) in inches.
        dpi : int, optional
            The resolution of the figure in dots per inch.
        xrange : tuple, optional
            The x-axis limits (xmin, xmax) to apply to all plots.
        styles : list of str, optional
            A list of matplotlib linestyles to apply to each label. Must
            match the length of `labels`.

        Returns
        -------
        matplotlib.axes.Axes or list of matplotlib.axes.Axes
            A single Axes object if one parameter is plotted, otherwise a
            list of Axes objects.
        """
        return plot_fit(
            self,
            labels=labels,
            which=which,
            figsize=figsize,
            dpi=dpi,
            xrange=xrange,
            styles=styles,
            *args,
            **kwargs,
        )

    def plot_comparison(
        self,
        label1: str,
        label2: str,
        which: Optional[List[str]] = None,
        figsize: Optional[tuple] = None,
        dpi: int = 150,
        xrange: Optional[tuple] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Overlays two fit results for direct visual comparison.

        This is a high-level wrapper around `plot_fit_comparison`.

        Parameters
        ----------
        label1 : str
            The label of the first fit to plot.
        label2 : str
            The label of the second fit to plot.
        which : list of str, optional
            A subset of fit parameters to plot. See `plot()` for options.
        figsize : tuple, optional
            Figure size in inches.
        dpi : int, optional
            Figure resolution in dots per inch.
        xrange : tuple, optional
            X-axis limits for all plots.

        Returns
        -------
        matplotlib.axes.Axes or list of matplotlib.axes.Axes
            A single or list of Axes objects.
        """
        return plot_fit_comparison(
            self,
            label1,
            label2,
            which=which,
            figsize=figsize,
            dpi=dpi,
            xrange=xrange,
            *args,
            **kwargs,
        )

    def plot_diff(
        self,
        label1: str,
        label2: str,
        which: Optional[List[str]] = None,
        figsize: Optional[tuple] = None,
        dpi: int = 150,
        xrange: Optional[tuple] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Plots the difference between two fit results.

        This is a high-level wrapper around `plot_fit_difference`.

        Parameters
        ----------
        label1 : str
            The label of the first fit (minuend).
        label2 : str
            The label of the second fit (subtrahend).
        which : list of str, optional
            A subset of fit parameters to plot. See `plot()` for options.
        figsize : tuple, optional
            Figure size in inches.
        dpi : int, optional
            Figure resolution in dots per inch.
        xrange : tuple, optional
            X-axis limits for all plots.

        Returns
        -------
        matplotlib.axes.Axes or list of matplotlib.axes.Axes
            A single or list of Axes objects.
        """
        return plot_fit_difference(
            self,
            label1,
            label2,
            which=which,
            figsize=figsize,
            dpi=dpi,
            xrange=xrange,
            *args,
            **kwargs,
        )

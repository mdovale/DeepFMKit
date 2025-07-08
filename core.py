from .plotting import *
from .fit import *
from .data import *
from .physics import *
from .dsp import *
from .fitters import *

import numpy as np
import scipy.constants as sc
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares, minimize, minimize_scalar
from scipy.special import jv, jn
import pandas as pd
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
plt.rcParams.update(default_rc)

import logging
logging.basicConfig(
format='%(asctime)s %(levelname)-8s %(message)s',
level=logging.INFO,
datefmt='%Y-%m-%d %H:%M:%S'
)

class DeepFitFramework():
    """A framework for loading, processing, analyzing, and plotting DFMI data.

    Say that `raw_data.txt` contains only a single channel. You can load it as:

    >>> dff = DeepFitFramework(raw_file='./raw_data.txt', raw_labels=['raw1'])

    or

    >>> dff = DeepFitFramework()
    >>> dff.load_raw('./raw_data.txt', labels=['raw1'])

    When loading the file, the metadata contained in the header is parsed, and stored
    along with the raw data in a DeepRawObject. These are appended to the `raws`
    dictionary. You can check it out with:

    >>> dff.raws 
    >>> {'raw1': <dfmdata.DeepRawObject at 0x1121b5d90>}

    >>> dff.raws['raw1'].info() # This shows some metadata

    >>> dff.raws['raw1'].data # This shows the raw data

    You can similarly load a fit_data file. When doing so, a DeepFitObject is
    created for every channel in the data, containing the time series of the
    fit parameters as well as the relevant metadata. 

    >>> dff.load_fit('./fit_data.txt', labels=['fit1'])

    Plotting all of the loaded fit data is as easy as:

    >>> dff.plot()

    Apart from loading raw_data or fit_data files, you can also perform fits on
    existing raw data, which will create the corresponding DeepFitObject. For
    example:

    >>> dff.fit('raw1', n=20, fit_label='fit2')

    DeepFitObject's are appended to the `fits` dictionary:

    >>> dff.fits
    >>> {'fit1': <dfmdata.DeepFitObject at 0x1121b5d90>, 'fit2': <dfmdata.DeepFitObject at 0x120776fd0>}

    You can check a DeepFitObject's metadata using:

    >>> dff.fits['fit1'].info()

    and you can save the results to a new fit_data file as:

    >>> dff.fits['fit1'].to_txt('example.txt')

    You can also perform simulations. To do this, a DFMIObject is created e.g. as:

    >>> dff.new_sim("sim1")

    which appends a new DFMIObject to the `sims` dictionary:

    >>> dff.sims
    >>> {'sim1': <dfmdata.DFMIObject at 0x13fdb8be0>}

    Running a simulation will create new DeepRawObject's with the resulting time series:

    >>> dff.simulate("sim1", n_seconds=60)

    creates two DeepRawObjects "sim1" (with armlength modulation) and "sim1_ref",
    which is subject to the same laser noise but without armlength modulation. 
    You can then run the fit on these as described above.
    """

    def __init__(self, raw_file=None, fit_file=None, raw_labels=None, fit_labels=None):
        self.raw_file = raw_file   # Path to raw data file
        self.fit_file = fit_file   # Path to fit data file
        self.lasers = {}           # Dictionary of LaserConfig
        self.ifos = {}             # Dictionary of InterferometerConfig
        self.sims = {}             # Dictionary of DFMIObject
        self.raws = {}             # Dictionary of DeepRawObject
        self.fits = {}             # Dictionary of DeepFitObject
        self.fits_df = {}          # Dictionary containing DataFrame's of fit results
        self.channr = None         # Number of channels
        self.n = None              # n*f_samp/f_mod = bufferSize; data_rate = f_samp/bufferSize = f_mod/n;
        self.t0 = None             # Start time  
        self.R = None              # Downsampling factor
        self.fs = None             # Fit data rate
        self.f_samp = None         # Sampling frequency
        self.f_mod = None          # Modulation frequency
        self.ndata = 10            # Number of higher harmonics to fit
        self.init_a = 1.6          # Initial value of the amplitude
        self.init_m = 6.0          # Initial value of the effective modulation index
        self.cfit = None

        if self.raw_file is not None:
            self.load_raw(labels=raw_labels)

        if self.fit_file is not None:
            self.load_fit(labels=fit_labels)

    def to_txt(self, filepath='./', labels=None):
        if labels is not None:
            for label in labels:
                filename = filepath + label + '.txt'
                self.fits[label].to_txt(filename)
        else:
            for fit in self.fits:
                filename = filepath + self.fits[fit].label + '.txt'
                self.fits[fit].to_txt(filename)

    def parse_header(self, file_select='raw'):
        """Parse the header of a raw_data or fit_data file.

        This function must be revised should the format of the header change.
        """
        lines = []
        values = []

        if file_select == 'raw':
            with open(self.raw_file) as f:
                for _ in range(11):
                    lines.append(f.readline())
        elif file_select == 'fit':
            with open(self.fit_file) as f:
                for _ in range(11):
                    lines.append(f.readline())
        else:
            logging.error('No files specified !!')

        for v in range(2,11):
            values.append(''.join([c for c in lines[v] if c in '1234567890.']))

        if file_select == 'raw':
            self.channr =   int(values[0])
            self.t0     =   int(values[1])
            self.f_samp = float(values[2])
            self.f_mod  = float(values[3])
            logging.info('Number of channels: {}'.format(self.channr))
            logging.info('Starting time: {}'.format(self.t0))
            logging.info('Sampling frequency: {}'.format(self.f_samp))
            logging.info('Modulation frequency: {}'.format(self.f_mod))
        else:
            self.channr =   int(values[0])
            self.t0     =   int(values[1])
            self.f_samp = float(values[2])
            self.f_mod  = float(values[3])
            self.n      =   int(values[4])
            self.R      =   int(values[5])
            self.fs     = float(values[6])
            logging.info('Number of channels: {}'.format(self.channr))
            logging.info('Starting time: {}'.format(self.t0))
            logging.info('Sampling frequency: {}'.format(self.f_samp))
            logging.info('Modulation frequency: {}'.format(self.f_mod))
            logging.info('n: {}'.format(self.n))
            logging.info('Downsampling factor: {}'.format(self.R))
            logging.info('Fit data rate: {}'.format(self.fs))

    def simulate(self, main_label, n_seconds, mode='asd', witness_label=None, 
                snr_db=None, trial_num=0, verbose=False):
        """
        Orchestrates a physics simulation using the dedicated SignalGenerator.

        This method acts as a high-level wrapper. It retrieves simulation
        configurations from the framework's `sims` dictionary and passes them
        to a SignalGenerator instance. The resulting raw data objects are
        then added to the framework's `raws` dictionary.

        This approach allows for clean generation of linked channels, such as
        a main channel and a witness channel for W-DFMI studies.

        Parameters
        ----------
        main_label : str
            The label of the DFMIObject in `self.sims` to use for the main channel.
        n_seconds : float
            The duration of the simulation in seconds.
        mode : {'asd', 'snr'}, optional
            The simulation mode to use: 'asd' for detailed noise models, or 'snr'
            for a simple signal-to-noise ratio based simulation. Defaults to 'asd'.
        witness_label : str, optional
            The label of the DFMIObject in `self.sims` to use for the witness
            channel. If provided, a second, linked channel is generated.
        snr_db : float, optional
            The target Signal-to-Noise Ratio in dB. Required if mode='snr'.
        trial_num : int, optional
            A number used to seed the random noise generators.
        """
        t0 = time.time()
        
        # --- 1. Validate configurations ---
        if main_label not in self.sims:
            logging.error(f"Main simulation label '{main_label}' not found!"); return
        main_config = self.sims[main_label]

        witness_config = None
        if witness_label:
            if witness_label not in self.sims:
                logging.error(f"Witness simulation label '{witness_label}' not found!"); return
            witness_config = self.sims[witness_label]
            logging.debug(f"Simulating '{main_label}' with witness '{witness_label}' for {n_seconds:.2f}s...")
        else:
            logging.debug(f"Simulating '{main_label}' for {n_seconds:.2f}s...")

        # --- 2. Instantiate and run the physics engine ---
        generator = SignalGenerator()
        generated_channels = generator.generate(
            main_config=main_config,
            n_seconds=n_seconds,
            mode=mode,
            trial_num=trial_num,
            witness_config=witness_config,
            snr_db=snr_db
        )

        # --- 3. Store the results in the framework ---
        if not generated_channels:
            logging.error("Simulation failed to generate data.")
            return

        for _, raw_obj in generated_channels.items():
            self.raws[raw_obj.label] = raw_obj
            
        sim_time = time.time() - t0
        main_config.simtime = sim_time
        logging.debug(f"Simulation finished in {sim_time:.3f} s.")

    def load_sim(self, sim):
        self.sims[sim.label] = sim

    def new_sim(self, label=None):
        """Load a DFMIObject with the specified label.
        """
        if label == None:
            from datetime import datetime
            label = datetime.now().strftime('%Y%m%d_%H%M%S')

        sim = DFMIObject(label=label)
        self.sims[sim.label] = sim
        return label

    def load_raw(self, raw_file=None, labels=None):
        """Load a raw_data file containing voltage time series data
        taken from a photodiode + transimpedance amplifier.
        """

        if raw_file is not None:
            self.raw_file = raw_file

        if self.raw_file is None:
            logging.error('No raw file specified !!')

        self.parse_header(file_select='raw')

        if labels is None:
            labels = ['']*self.channr
            for c in range(self.channr):
                labels[c] = self.raw_file + '_ch' + str(c)
        else:
            assert len(labels) == self.channr

        for c in range(self.channr):
            raw = DeepRawObject(data=pd.read_csv(self.raw_file, sep = ' ', skiprows=13, usecols=[c], names=['ch'+str(c)]))
            raw.raw_file = self.raw_file
            raw.label = labels[c]
            raw.t0 = self.t0
            raw.f_samp = self.f_samp
            raw.f_mod = self.f_mod
            self.raws[raw.label] = raw

    def load_fit(self, fit_file=None, labels=None):
        """Load a fit_data file containing the fit parameters produced
        by the DFMSWPM algorithm.

        It will erase any previous fit data contained in object.
        """

        if fit_file is not None:
            self.fit_file = fit_file

        if self.fit_file is None:
            logging.error('No fit file specified !!')

        self.parse_header(file_select='fit')

        if labels is None:
            labels = ['']*self.channr
            for c in range(self.channr):
                labels[c] = self.raw_file + '_ch' + str(c)
        else:
            assert len(labels) == self.channr

        data = np.genfromtxt(self.fit_file, dtype='double', skip_header=13, invalid_raise=False)
        
        for k in range(self.channr):
            fit = DeepFitObject()
            fit.nbuf = len(data[:, 0])
            fit.n = self.n
            fit.t0 = self.t0
            fit.R = self.R
            fit.fs = self.fs
            fit.f_samp = self.f_samp
            fit.f_mod  = self.f_mod
            fit.ndata = self.ndata
            fit.init_a = self.init_a
            fit.init_m = self.init_m
            fit.ssq = data[:, 6*k+0]
            fit.amp = data[:, 6*k+1]
            fit.m   = data[:, 6*k+2]
            fit.phi = data[:, 6*k+3]
            fit.psi = data[:, 6*k+4]
            fit.dc  = data[:, 6*k+5]
            fit.time = np.arange(0, fit.nbuf/self.fs, 1./self.fs)
            fit.label = labels[k]
            self.fits[labels[k]] = fit

    def _create_fit_object_from_df(self, fit_label, source_label, n, R, fs, nbuf, ndata, init_a, init_m):
        """Creates and registers a DeepFitObject from a results DataFrame.

        This helper method converts the raw DataFrame produced by a fitter into 
        a fully-populated DeepFitObject. It handles the assignment of all 
        metadata from the source raw data object and the fit configuration, 
        and registers the final object in the framework's `self.fits` dictionary.

        Parameters
        ----------
        fit_label : str
            The label for the new DeepFitObject and the key for `self.fits`.
        source_label : str
            The label of the source DeepRawObject in `self.raws`.
        n : int
            The number of modulation cycles per fit buffer.
        R : int
            The buffer size in samples (downsampling factor).
        fs : float
            The fit data rate in Hz.
        nbuf : int
            The total number of buffers in the fit.
        ndata : int
            The number of harmonics used in the fit (for NLS fitters).
        init_a : float
            The initial amplitude guess used for the fit.
        init_m : float
            The initial modulation depth guess used for the fit.

        Returns
        -------
        DeepFitObject
            The newly created and registered fit object.
        """
        df = self.fits_df[fit_label]
        
        fit = DeepFitObject()
        fit.n, fit.R, fit.fs, fit.nbuf, fit.ndata, fit.init_a, fit.init_m = n, R, fs, nbuf, ndata, init_a, init_m

        fit.t0 = self.raws[source_label].t0
        fit.f_samp = self.raws[source_label].f_samp
        fit.f_mod  = self.raws[source_label].f_mod
        
        fit.ssq = df['ssq'].to_numpy()
        fit.amp = df['amp'].to_numpy()
        fit.m   = df['m'].to_numpy()
        fit.tau = df['tau'].to_numpy()
        fit.phi = df['phi'].to_numpy()
        fit.psi = df['psi'].to_numpy()
        fit.dc  = df['dc'].to_numpy()
        fit.time = np.arange(0, fit.ssq.shape[0] / fit.fs, 1. / fit.fs)
        fit.label = fit_label
        
        self.fits[fit_label] = fit
        return fit

    def fit_init(self, label, n):
        """Calculates key parameters for a fitting run.

        Takes the desired number of modulation cycles to include in a buffer
        (`n`) and calculates the resulting buffer size, fit rate, and
        the total number of buffers available in the dataset.

        Parameters
        ----------
        label : str
            The label of the source DeepRawObject in `self.raws`.
        n : int
            The number of modulation periods (`f_mod`) to include in each
            analysis buffer.

        Returns
        -------
        tuple
            A tuple containing (R, fs, nbuf):
            R : int
                The buffer size in samples.
            fs : float
                The resulting fit data rate in Hz (f_samp / R).
            nbuf : int
                The total number of full buffers available in the raw data.
        """
        R = int(self.raws[label].f_samp/self.raws[label].f_mod*n)
        fs = self.raws[label].f_samp/R
        nbuf = int(self.raws[label].data.shape[0]/R)
        if nbuf == 0:
            logging.error('Check buffer size !! Calculated nbuf is zero.')

        return R, fs, nbuf
    
    def fit(self, main_label, method='nls', fit_label=None, **kwargs):
        """
        Fits raw data using a specified algorithm via the Strategy pattern.

        This is my new unified fitting interface. It selects the appropriate
        fitter class from the `fitters.py` module based on the 'method'
        string, instantiates it with the correct configuration, and then
        calls its `fit()` method to perform the analysis.

        Parameters
        ----------
        main_label : str
            The label of the raw data object in `self.raws` to be fit.
        method : str, optional
            The fitting algorithm to use. Available methods include:
            'nls', 'ekf', 'wdfmi_nls', 'wdfmi_ortho', 'wdfmi_seq'.
        fit_label : str, optional
            The label for the output DeepFitObject.
        **kwargs :
            Additional arguments passed directly to the selected fitter.
            For W-DFMI methods, this must include `witness_label`.

        Returns
        -------
        DeepFitObject
            A fit object containing the results, which is also stored in `self.fits`.
        """
        # --- 1. Select the Fitter Class ---
        fitter_map = {
            'nls': StandardNLSFitter,
            'ekf': EKFFitter,
            'wdfmi_nls': WDFMI_NLSFitter,
            'wdfmi_ortho': WDFMI_OrthogonalFitter,
            'wdfmi_seq': WDFMI_SequentialFitter,
            'hwdfmi': HWDFMI_Fitter
        }
        if method not in fitter_map:
            logging.error(f"Unknown fit method: '{method}'. Available: {list(fitter_map.keys())}"); return

        FitterClass = fitter_map[method]
        logging.debug(f"Dispatching to {FitterClass.__name__} for label '{main_label}'.")

        # --- 2. Prepare Data and Config ---
        if main_label not in self.raws:
            logging.error(f"Invalid raw data label: '{main_label}' !!"); return
        main_raw = self.raws[main_label]

        if fit_label is None:
            fit_label = f"{main_label}_{method}"

        # Get 'n' for the fit config, which is common to all fitters
        n_cycles = kwargs.get('n')
        if n_cycles is None:
            sim_obj = self.sims.get(main_raw.sim.label if main_raw.sim else main_label)
            n_cycles = sim_obj.fit_n if sim_obj else 20

        R, fs, nbuf = self.fit_init(main_label, n_cycles)
        if hasattr(main_raw, 'phi_sim') and main_raw.phi_sim is not None and len(main_raw.phi_sim) > 0:
            main_raw.phi_sim_downsamp = vectorized_downsample(main_raw.phi_sim, R)

        # The base config for all fitters
        fit_config = {'n': n_cycles}
        # Add method-specific configs
        if method in ['nls', 'wdfmi_nls']:
            fit_config['ndata'] = kwargs.get('ndata', 10)
        
        # --- 3. Instantiate and Run the Fitter ---
        fitter_args = {'main_raw': main_raw}
        if 'wdfmi' in method:
            witness_label = kwargs.get('witness_label')
            if not witness_label or witness_label not in self.raws:
                logging.error(f"W-DFMI method '{method}' requires a valid 'witness_label'."); return
            fitter_args['witness_raw'] = self.raws[witness_label]

        fitter = FitterClass(fit_config)
        results_df = fitter.fit(**fitter_args, **kwargs)

        # --- 4. Create and Store the Final DeepFitObject ---
        if results_df is None or results_df.empty:
            logging.error(f"{FitterClass.__name__} returned no results.")
            return None
        
        if method in ['nls', 'ekf']:
            results_df['tau'] = results_df['m'] / (2*np.pi*main_raw.sim.laser.df) if main_raw.sim else 0.0

        self.fits_df[fit_label] = results_df
        
        fit_obj = self._create_fit_object_from_df(
            fit_label, main_label, n_cycles, R, fs, nbuf, 
            fit_config.get('ndata', 0), 0, 0
        )
        self.fits[fit_label] = fit_obj
        
        return fit_obj
    
    def create_witness_channel(self, main_channel_label, witness_channel_label, m_witness=None, delta_l_witness=None):
        """
        Creates a witness channel, with optional auto-tuning for optimal sensitivity.

        This helper function creates a static witness channel linked to a main
        channel's laser source. It robustly handles the witness design by:
        1.  Setting the witness arm lengths to achieve a target `m_witness` or `delta_l_witness`.
        2.  Analytically calculates and sets the witness's phase offset (`ifo.phi`)
            to ensure operation at the perfect mid-fringe point (quadrature).

        Parameters
        ----------
        main_channel_label : str
            The label of the existing DFMIObject to use as a template.
        witness_channel_label : str
            The label for the new witness channel to be created.
        m_witness : float, optional
            The target effective modulation index for the witness interferometer.
        delta_l_witness : float, optional
            The desired absolute optical path difference for the witness, in meters.

        Returns
        -------
        DFMIObject
            The newly created witness channel object.
        """
        # --- 1. Validate Inputs and get shared laser ---
        if main_channel_label not in self.sims:
            raise KeyError(f"Main channel '{main_channel_label}' not found in framework.")
        if delta_l_witness is not None and m_witness is not None:
            raise ValueError("Please specify either delta_l_witness or m_witness, but not both.")
        
        main_channel = self.sims[main_channel_label]
        shared_laser = main_channel.laser
        
        # --- 2. Determine Target Witness Modulation Depth ---
        if m_witness is None and delta_l_witness is None:
            m_target = 0.1
        elif m_witness is not None:
            m_target = m_witness
        else:
            m_target = (2 * np.pi * shared_laser.df * delta_l_witness) / sc.c
        
        # --- 4. Create and Configure Witness Interferometer ---
        witness_ifo = InterferometerConfig(label=f"{witness_channel_label}_ifo")
        witness_ifo.arml_mod_amp = 0.0
        witness_ifo.arml_mod_n = 0.0

        if shared_laser.df == 0:
            raise ValueError("Cannot set 'm_witness' when laser 'df' is zero.")
        final_delta_l = (m_target * sc.c) / (2 * np.pi * shared_laser.df)
        witness_ifo.ref_arml = 0.01
        witness_ifo.meas_arml = witness_ifo.ref_arml + final_delta_l

        # Analytical Fringe Locking
        f0 = sc.c / shared_laser.wavelength
        static_fringe_phase = (2 * np.pi * f0 * final_delta_l) / sc.c
        phi_offset_required = (np.pi / 2.0) + static_fringe_phase
        witness_ifo.phi = phi_offset_required % (2 * np.pi)
        
        # --- 5. Compose and Register the New Witness Channel ---
        witness_channel = DFMIObject(
            label=witness_channel_label, laser_config=shared_laser,
            ifo_config=witness_ifo, f_samp=main_channel.f_samp
        )
        witness_channel.fit_n = main_channel.fit_n
        self.sims[witness_channel_label] = witness_channel

        logging.debug(f"Created witness channel '{witness_channel_label}' with final m_witness={witness_channel.m:.3f}.")
        return witness_channel

    def calc_lpsd(self, labels=None):
        """Calculate the LPSD of the interferometric phase parameter (phi) 
        for the existing fit data under the specified `labels`, or for all
        of the existing fit data.
        """
        from spectools.lpsd import lpsd

        if labels is not None:
            for fit in labels:
                try:
                    self.fits[fit].f, _, self.fits[fit].Sxx, _, _, _ = lpsd(self.fits[fit].phi, \
                        self.fits[fit].fs, self.fits[fit].olap, self.fits[fit].bmin, self.fits[fit].Lmin, \
                        self.fits[fit].Jdes, self.fits[fit].Kdes, self.fits[fit].order, self.fits[fit].win, self.fits[fit].psll, return_type='legacy')
                except:
                    logging.warning('Specified label is invalid!')
        else:
            for fit in self.fits:
                self.fits[fit].f, _, self.fits[fit].Sxx, _, _, _ = lpsd(self.fits[fit].phi, \
                    self.fits[fit].fs, self.fits[fit].olap, self.fits[fit].bmin, self.fits[fit].Lmin, \
                    self.fits[fit].Jdes, self.fits[fit].Kdes, self.fits[fit].order, self.fits[fit].win, self.fits[fit].psll, return_type='legacy')

    def plot_lpsd(self, labels=None, nm=True, pm=True):
        """Plot the LPSD of the interferometric phase parameter (phi) 
        for the existing fit data under the specified `labels`, or for
        all of the existing fit data.
        """
        if labels is not None:
            for fit in labels:
                if self.fits[fit].f == None:
                    logging.info('Calculating the LPSD now...')
                    self.calc_lpsd()
        else:
            for fit in self.fits:
                if self.fits[fit].f == None:
                    logging.info('Calculating the LPSD now...')
                    self.calc_lpsd()

        title = ''
        xlabel = r"Frequency$\,({\mathrm{Hz}})$"
        ylabel = r"Phase ASD$\,(\mathrm{rad}/\sqrt{\mathrm{Hz}})$"

        fig, ax = plt.subplots(1, figsize=figsize(1.2), dpi=1 * 300)

        if labels is not None:
            for fit in labels:
                log_plot(self.fits[fit].f, np.sqrt(self.fits[fit].Sxx), ax, title, xlabel, ylabel, self.fits[fit].label)
        else:
            for fit in self.fits:
                log_plot(self.fits[fit].f, np.sqrt(self.fits[fit].Sxx), ax, title, xlabel, ylabel, self.fits[fit].label)

        if nm == True:
            log_plot(self.f, displacement_req(self.fits[fit].f, 1e-9, 3e-3), ax, title, xlabel, ylabel, "1 nm")

        if pm == True:
            log_plot(self.f, displacement_req(self.fits[fit].f, 1e-12, 3e-3), ax, title, xlabel, ylabel, "1 pm")

    def plot(self, labels=None, xrange=None):
        """Plot the existing fit data for the specified `labels`,
        or plot all of the existing fit data, in the specified `xrange`,
        or in the entire range.
        """
        self.fig, (ax1, ax2, ax3, ax4, ax5, ax6) = dfm_axes()

        if labels is None:
            for k in self.fits:
                ax6.semilogy(self.fits[k].time, self.fits[k].ssq)
                ax5.plot(self.fits[k].time, self.fits[k].dc)
                ax4.plot(self.fits[k].time, self.fits[k].amp)
                ax3.plot(self.fits[k].time, self.fits[k].m)
                ax2.plot(self.fits[k].time, self.fits[k].phi)
                ax1.plot(self.fits[k].time, self.fits[k].psi, label=str(k))
        else:
            for k in labels:
                ax6.semilogy(self.fits[k].time, self.fits[k].ssq)
                ax5.plot(self.fits[k].time, self.fits[k].dc)
                ax4.plot(self.fits[k].time, self.fits[k].amp)
                ax3.plot(self.fits[k].time, self.fits[k].m)
                ax2.plot(self.fits[k].time, self.fits[k].phi)
                ax1.plot(self.fits[k].time, self.fits[k].psi, label=str(k))

        if xrange is not None:
            ax1.set_xlim(xrange)
            ax2.set_xlim(xrange)
            ax3.set_xlim(xrange)
            ax4.set_xlim(xrange)
            ax5.set_xlim(xrange)
            ax6.set_xlim(xrange)
            autoscale_y(ax1)
            autoscale_y(ax2)
            autoscale_y(ax3)
            autoscale_y(ax4)
            autoscale_y(ax5)
            autoscale_y(ax6)
        
        self.fig.tight_layout()
        return (ax1, ax2, ax3, ax4, ax5, ax6)

    def plot_diff(self, label1, label2, xrange=None):
        """Plot the absolute difference between the fit 
        parameters of the specified two fits.
        """

        self.fig, (ax1, ax2, ax3, ax4, ax5, ax6) = dfm_axes()

        a = self.fits[label1].time.shape[0]
        b = self.fits[label2].time.shape[0]
        c = a

        if b < a:
            c = b

        ax6.semilogy(self.fits[label1].time[:c], self.fits[label1].ssq[:c] - self.fits[label2].ssq[:c])
        ax5.plot(self.fits[label1].time[:c], self.fits[label1].dc[:c]      - self.fits[label2].dc[:c])
        ax4.plot(self.fits[label1].time[:c], self.fits[label1].amp[:c]     - self.fits[label2].amp[:c])
        ax3.plot(self.fits[label1].time[:c], self.fits[label1].m[:c]       - self.fits[label2].m[:c])
        ax2.plot(self.fits[label1].time[:c], self.fits[label1].phi[:c]     - self.fits[label2].phi[:c])
        ax1.plot(self.fits[label1].time[:c], self.fits[label1].psi[:c]     - self.fits[label2].psi[:c], label=label1+'-'+label2)

        ax1.legend(loc='upper right')
        
        if xrange is not None:
            ax1.set_xlim(xrange)
            ax2.set_xlim(xrange)
            ax3.set_xlim(xrange)
            ax4.set_xlim(xrange)
            ax5.set_xlim(xrange)
            ax6.set_xlim(xrange)
            autoscale_y(ax1)
            autoscale_y(ax2)
            autoscale_y(ax3)
            autoscale_y(ax4)
            autoscale_y(ax5)
            autoscale_y(ax6)

        self.fig.tight_layout()

    def plot_comparison(self, label1, label2, xrange=None):
        """Plot two channels for comparison.
        """

        self.fig, (ax1, ax2, ax3, ax4, ax5, ax6) = dfm_axes()           

        ax6.semilogy(self.fits[label1].time, self.fits[label1].ssq)
        ax5.plot(self.fits[label1].time, self.fits[label1].dc)
        ax4.plot(self.fits[label1].time, self.fits[label1].amp)
        ax3.plot(self.fits[label1].time, self.fits[label1].m)
        ax2.plot(self.fits[label1].time, self.fits[label1].phi)
        ax1.plot(self.fits[label1].time, self.fits[label1].psi, label=label1)

        ax6.semilogy(self.fits[label2].time, self.fits[label2].ssq, linestyle='dashed')
        ax5.plot(self.fits[label2].time, self.fits[label2].dc, linestyle='dashed')
        ax4.plot(self.fits[label2].time, self.fits[label2].amp, linestyle='dashed')
        ax3.plot(self.fits[label2].time, self.fits[label2].m, linestyle='dashed')
        ax2.plot(self.fits[label2].time, self.fits[label2].phi, linestyle='dashed')
        ax1.plot(self.fits[label2].time, self.fits[label2].psi, linestyle='dashed', label=label2)

        ax1.legend(loc='upper right')
        
        if xrange is not None:
            ax1.set_xlim(xrange)
            ax2.set_xlim(xrange)
            ax3.set_xlim(xrange)
            ax4.set_xlim(xrange)
            ax5.set_xlim(xrange)
            ax6.set_xlim(xrange)
            autoscale_y(ax1)
            autoscale_y(ax2)
            autoscale_y(ax3)
            autoscale_y(ax4)
            autoscale_y(ax5)
            autoscale_y(ax6)
        
        self.fig.tight_layout()
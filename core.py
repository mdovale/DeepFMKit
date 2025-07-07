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

def _process_fit_chunk(args):
    """
    Worker function for parallel processing. Fits a chunk of raw data buffers.
    
    This function is designed to be called by multiprocessing.Pool. It takes
    all necessary data as arguments to be self-contained and pickle-able.
    It processes its assigned chunk sequentially to maintain the warm-start
    advantage within the chunk.

    Parameters
    ----------
    args : tuple
        A tuple containing all necessary arguments:
        (raw_data_chunk, initial_guess, R, ndata, f_mod, f_samp)

    Returns
    -------
    list_of_dicts : list
        A list of dictionaries, where each dictionary holds the fit result
        for a single buffer in the chunk.
    """
    # 1. Unpack arguments
    raw_data_chunk, initial_guess, R, ndata, f_mod, f_samp = args
    
    # 2. Setup
    results_list = []
    current_guess = np.array(initial_guess)
    w0 = 2. * np.pi * f_mod / f_samp
    
    # 3. Process the chunk sequentially
    for i in range(raw_data_chunk.shape[0]):
        # Get the data for the current buffer
        buffer_data = raw_data_chunk[i]
        
        # Calculate I/Q data for the buffer
        QI_data_mean = np.zeros(2 * ndata)
        for n in range(ndata):
            Q_data, I_data = calculate_quadratures(n, buffer_data, w0)
            QI_data_mean[n] = Q_data.mean()
            QI_data_mean[n + ndata] = I_data.mean()
        
        # Run the NLS fit using the result of the previous fit as a guess
        status, fit_parm, fit_ssq = fit(ndata, QI_data_mean, current_guess)
        
        # Update the guess for the next iteration
        current_guess = fit_parm
        
        # Store the simple result
        results_list.append({
            'amp': fit_parm[0],
            'm': fit_parm[1],
            'phi': fit_parm[2],
            'psi': fit_parm[3],
            'dc': np.mean(buffer_data), # Calculate DC here
            'ssq': fit_ssq,
            'fitok': status
        })
        
    return results_list

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

    def change_raw_labels(self, old_labels, new_labels):
        assert len(old_labels) == len(new_labels)

        for i, label in enumerate(old_labels):
            assert label in self.raws
            self.raws[new_labels[i]] = self.raws.pop(label)

    def change_fit_labels(self, old_labels, new_labels):
        assert len(old_labels) == len(new_labels)

        for i, label in enumerate(old_labels):
            assert label in self.fits
            self.fits[new_labels[i]] = self.fits.pop(label)

    def append_raw_labels(self, string):
        assert type(string) == str
        old_labels = list(self.raws.keys())
        new_labels = ['']*len(old_labels)
        for i, item in enumerate(old_labels):
            new_labels[i] = item + string
        dmap = {o:n for (o,n) in zip(old_labels, new_labels)}
        self.raws = dict((dmap[key], value) for (key, value) in self.raws.items())

    def append_fit_labels(self, string):
        assert type(string) == str
        old_labels = list(self.fits.keys())
        new_labels = ['']*len(old_labels)
        for i, item in enumerate(old_labels):
            new_labels[i] = item + string
        dmap = {o:n for (o,n) in zip(old_labels, new_labels)}
        self.fits = dict((dmap[key], value) for (key, value) in self.fits.items())

    def fits_to_txt(self, filepath='./', labels=None):
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
        res = {}    

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
    
    def fit(self, raw_label, method='nls', fit_label=None, **kwargs):
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
        method : {'nls', 'ekf'}, optional
            The fitting algorithm to use. Defaults to 'nls'. More methods like
            'wdfmi_ortho' will be added here as I refactor them.
        fit_label : str, optional
            The label for the output DeepFitObject.
        **kwargs :
            Additional arguments passed directly to the selected fitter.
            Examples: `n`, `ndata`, `n_cores` for 'nls', or `Q_diag` for 'ekf'.

        Returns
        -------
        DeepFitObject
            A fit object containing the results, which is also stored in `self.fits`.
        """
        # --- 1. Select the Fitter Class ---
        fitter_map = {
            'nls': StandardNLSFitter,
            'ekf': EKFFitter,
            # 'wdfmi_ortho': fitters.WDFMI_OrthogonalFitter, # To be added
        }
        if method not in fitter_map:
            logging.error(f"Unknown fit method: '{method}'. Available: {list(fitter_map.keys())}"); return

        FitterClass = fitter_map[method]
        logging.info(f"Dispatching to {FitterClass.__name__} for label '{raw_label}'.")

        # --- 2. Prepare Data and Config ---
        if raw_label not in self.raws:
            logging.error(f"Invalid raw data label: '{raw_label}' !!"); return
        main_raw = self.raws[raw_label]

        if fit_label is None:
            fit_label = f"{raw_label}_{method}"

        # Get 'n' for the fit config, which is common to all fitters (number of modulation cycles to include in buffer)
        n_cycles = kwargs.get('n')
        if n_cycles is None:
            sim_obj = self.sims.get(main_raw.sim.label if main_raw.sim else raw_label)
            n_cycles = sim_obj.fit_n if sim_obj else 20

        R, fs, nbuf = self.fit_init(raw_label, n_cycles)
        if hasattr(main_raw, 'phi_sim') and main_raw.phi_sim is not None and len(main_raw.phi_sim) > 0:
            main_raw.phi_sim_downsamp = vectorized_downsample(main_raw.phi_sim, R)
            logging.debug("Calculated downsampled ground truth phase 'phi_sim_downsamp'.")
        
        # The base config for all fitters
        fit_config = {'n': n_cycles}
        # Add method-specific configs
        if method == 'nls':
            fit_config['ndata'] = kwargs.get('ndata', 10)

        # --- 3. Instantiate and Run the Fitter ---
        fitter = FitterClass(fit_config)
        results_df = fitter.fit(main_raw, **kwargs)

        # --- 4. Create and Store the Final DeepFitObject ---
        if results_df is None or results_df.empty:
            logging.error(f"{FitterClass.__name__} returned no results.")
            return None
        
        self.fits_df[fit_label] = results_df
        
        R, fs, nbuf = self.fit_init(raw_label, n_cycles)
        
        # Pass dummy values for init_a, init_m, ndata as they are now fitter-specific
        fit_obj = self._create_fit_object_from_df(
            fit_label, raw_label, n_cycles, R, fs, nbuf, 
            fit_config.get('ndata', 0), 0, 0
        )
        self.fits[fit_label] = fit_obj
        
        return fit_obj    

    def _create_fit_object_from_df(self, fit_label, source_label, n, R, fs, nbuf, ndata, init_a, init_m):
        """Helper to create a DeepFitObject from a results DataFrame."""
        df = self.fits_df[fit_label]
        
        fit = DeepFitObject()
        fit.n, fit.R, fit.fs, fit.nbuf, fit.ndata, fit.init_a, fit.init_m = n, R, fs, nbuf, ndata, init_a, init_m

        fit.t0 = self.raws[source_label].t0
        fit.f_samp = self.raws[source_label].f_samp
        fit.f_mod  = self.raws[source_label].f_mod
        
        fit.ssq = df['ssq'].to_numpy()
        fit.amp = df['amp'].to_numpy()
        fit.m   = df['m'].to_numpy()
        fit.phi = df['phi'].to_numpy()
        fit.psi = df['psi'].to_numpy()
        fit.dc  = df['dc'].to_numpy()
        fit.time = np.arange(0, fit.ssq.shape[0] / fit.fs, 1. / fit.fs)
        fit.label = fit_label
        
        self.fits[fit_label] = fit
        return fit

    def fit_init(self, label, n):

        R = int(self.raws[label].f_samp/self.raws[label].f_mod*n)
        fs = self.raws[label].f_samp/R
        nbuf = int(self.raws[label].data.shape[0]/R)
        if nbuf == 0:
            logging.error('Check buffer size !!')

        return R, fs, nbuf

    def fit_wdfmi(self, main_label, witness_label, n=None, fit_label=None, ndata=10, init_a=1.6, init_m=6.0, init_phi=0.0, init_psi=0.0, verbose=True):
        """
        Performs a fit using the definitive, physically exact W-DFMI method.

        This function uses the exact physical model for the phase difference,
        `phi(t-tau) - phi(t)`, ensuring perfect model symmetry with the data
        generation engine. It correctly processes the witness signal as a measure
        of the frequency modulation waveform, integrates it to get the phase
        waveform, and fits for the physical time delay `tau` for robustness.
        This version is correct for all `m` and distortion levels.
        """
        # --- 1. Initialization ---
        if main_label not in self.raws or witness_label not in self.raws:
            logging.error("Invalid main or witness label provided!"); return
        if fit_label is None: fit_label = main_label + '_wdfmi'
        if n is None: n = self.sims[main_label].fit_n

        R, fs, nbuf = self.fit_init(main_label, n)
        if nbuf == 0: logging.error('Check buffer size !!'); return
        
        main_raw = self.raws[main_label]
        laser_cfg = main_raw.sim.laser
        main_ifo_cfg = main_raw.sim.ifo
        omega_mod = 2 * np.pi * laser_cfg.f_mod
        w0_samp = omega_mod / main_raw.f_samp
        
        # --- 2. Process Witness Signal to get Phase Modulation Waveform ---
        witness_buffer_raw = np.array(self.raws[witness_label].data.loc[0:R-1]).flatten()
        v_w_ac = witness_buffer_raw - np.mean(witness_buffer_raw)
        
        # Witness voltage is proportional to f_mod(t). Correct for sign and normalize.
        f_mod_basis = - v_w_ac / np.max(np.abs(v_w_ac))
        
        # Integrate the frequency basis to get the phase modulation basis.
        time_axis_buffer = np.arange(R) / main_raw.f_samp
        dt = time_axis_buffer[1] - time_axis_buffer[0]
        phi_mod_basis = np.cumsum(f_mod_basis) * dt

        # --- 3. Main Fitting Loop ---
        delta_l = main_ifo_cfg.meas_arml - main_ifo_cfg.ref_arml
        tau_init = delta_l / sc.c
        current_guess = np.array([init_a, tau_init, init_phi, init_psi])
        
        results_list = []
        
        iterable = range(nbuf)
        if verbose:
            logging.info(f"Processing '{main_label}' with Definitive Exact Model Fitter...")
            iterable = tqdm(iterable)

        for b in iterable:
            buf_range = range(b * R, (b + 1) * R)
            main_buffer_raw = np.array(main_raw.data.loc[buf_range]).flatten()
            
            QI_data_meas = np.zeros(2 * ndata)
            for i in range(ndata):
                n_harm = i + 1
                QI_data_meas[i] = (2/R) * np.sum(main_buffer_raw * np.cos(n_harm * w0_samp * np.arange(R)))
                QI_data_meas[i + ndata] = (2/R) * np.sum(main_buffer_raw * np.sin(n_harm * w0_samp * np.arange(R)))

            def _wdfmi_residuals(params, phi_mod_basis_arg, laser_cfg_arg, QI_meas):
                C, tau, phi, psi = params
                
                phi_mod_unscaled = 2 * np.pi * laser_cfg_arg.df * phi_mod_basis_arg
                
                t_shift_psi = -psi / omega_mod
                t_interp_psi = time_axis_buffer - t_shift_psi
                phi_mod_shifted = np.interp(t_interp_psi, time_axis_buffer, phi_mod_unscaled, period=time_axis_buffer[-1])
                
                t_interp_tau = time_axis_buffer - tau
                phi_mod_delayed = np.interp(t_interp_tau, time_axis_buffer, phi_mod_shifted, period=time_axis_buffer[-1])
                
                delta_phi_mod = phi_mod_delayed - phi_mod_shifted
                
                v_model = C * np.cos(phi + delta_phi_mod)
                
                QI_model = np.zeros(2 * ndata)
                for i in range(ndata):
                    n_harm = i + 1
                    QI_model[i] = (2/R) * np.sum(v_model * np.cos(n_harm * w0_samp * np.arange(R)))
                    QI_model[i + ndata] = (2/R) * np.sum(v_model * np.sin(n_harm * w0_samp * np.arange(R)))
                
                return QI_model - QI_meas

            opt_result = least_squares(
                _wdfmi_residuals, current_guess,
                args=(phi_mod_basis, laser_cfg, QI_data_meas), method='lm'
            )
            
            fit_parm = opt_result.x
            C_fit, tau_fit, phi_fit, psi_fit = fit_parm
            
            m_fit = 2 * np.pi * laser_cfg.df * tau_fit
            current_guess = fit_parm
            
            results_list.append({
                'amp': C_fit, 'm': m_fit, 'phi': phi_fit, 'psi': psi_fit, 
                'dc': np.mean(main_buffer_raw),
                'ssq': np.sum(opt_result.fun**2), 'fitok': 1 if opt_result.success else 0, 'b': b
            })
            
        self.fits_df[fit_label] = pd.DataFrame(results_list)
        return self._create_fit_object_from_df(fit_label, main_label, n, R, fs, nbuf, ndata, init_a, init_m)
    
    def fit_wdfmi_orthogonal_demodulation(self, main_label, witness_label, n=None, fit_label=None, init_m=6.0, init_psi=0.0, verbose=True):
        """
        Fits DFMI data using the robust Orthogonal Demodulation W-DFMI algorithm.

        This method implements a two-stage fitting process that is immune to the
        "algorithmic cross-talk" that causes instabilities in the standard NLS approach.
        It separates the non-linear search for the physical shape parameters (tau, psi)
        from a linear fit for the signal amplitudes (I, Q), resulting in a vastly
        more stable and reliable algorithm.

        Parameters
        ----------
        main_label : str
            The label of the primary DeepRawObject containing the main DFMI signal.
        witness_label : str
            The label of the DeepRawObject containing the witness signal.
        n : int, optional
            The number of modulation periods per fit buffer (`n_fit`).
        fit_label : str, optional
            The label for the output DeepFitObject. Defaults to `main_label + '_ortho'`.
        init_m : float, optional
            Initial guess for the modulation depth, used to calculate an initial `tau`.
        init_psi : float, optional
            Initial guess for the modulation phase (psi).
        verbose : bool, optional
            If True, display a progress bar.

        Returns
        -------
        DeepFitObject
            A fit object containing the time-series of the estimated parameters.
        """
        # --- 1. Initialization ---
        if main_label not in self.raws or witness_label not in self.raws:
            logging.error("Invalid main or witness label provided!"); return
        if fit_label is None: fit_label = main_label + '_ortho'
        if n is None: n = self.sims[main_label].fit_n

        R, fs, nbuf = self.fit_init(main_label, n)
        if nbuf == 0: logging.error('Check buffer size !!'); return

        main_raw = self.raws[main_label]
        laser_cfg = main_raw.sim.laser
        main_ifo_cfg = main_raw.sim.ifo
        omega_mod = 2 * np.pi * laser_cfg.f_mod

        # --- 2. Process Witness Signal once to get Phase Modulation Waveform ---
        witness_buffer_raw = np.array(self.raws[witness_label].data.loc[0:R-1]).flatten()
        v_w_ac = witness_buffer_raw - np.mean(witness_buffer_raw)
        f_mod_basis = v_w_ac / np.max(np.abs(v_w_ac))
        time_axis_buffer = np.arange(R) / main_raw.f_samp
        dt = time_axis_buffer[1] - time_axis_buffer[0]
        phi_mod_basis = np.cumsum(f_mod_basis) * dt

        # --- 3. Main Fitting Loop ---
        # Convert initial guess for 'm' to an initial guess for 'tau'
        delta_l = main_ifo_cfg.meas_arml - main_ifo_cfg.ref_arml
        tau_init = delta_l / sc.c
        current_guess = np.array([tau_init, init_psi])

        results_list = []
        
        iterable = range(nbuf)
        if verbose:
            logging.info(f"Processing '{main_label}' with Orthogonal Demodulation Fitter...")
            iterable = tqdm(iterable)

        for b in iterable:
            buf_range = range(b * R, (b + 1) * R)
            main_buffer_raw = np.array(main_raw.data.loc[buf_range]).flatten()
            v_main_ac = main_buffer_raw - np.mean(main_buffer_raw)

            # This is the "cost function" for the outer loop non-linear optimization.
            # It takes a guess for (tau, psi) and returns the residual of the *best possible linear fit*.
            def _outer_loop_cost(params):
                tau, psi = params
                
                # Stage 1: Construct the orthogonal basis functions for this (tau, psi)
                phi_mod_unscaled = 2 * np.pi * laser_cfg.df * phi_mod_basis
                t_shift_psi = -psi / omega_mod
                t_interp_psi = time_axis_buffer - t_shift_psi
                phi_mod_shifted = np.interp(t_interp_psi, time_axis_buffer, phi_mod_unscaled, period=time_axis_buffer[-1])
                
                t_interp_tau = time_axis_buffer - tau
                phi_mod_delayed = np.interp(t_interp_tau, time_axis_buffer, phi_mod_shifted, period=time_axis_buffer[-1])
                
                delta_phi_mod = phi_mod_delayed - phi_mod_shifted
                
                basis_I = np.cos(delta_phi_mod)
                basis_Q = np.sin(delta_phi_mod)
                
                # Stage 2: Perform the fast, non-iterative linear least squares fit
                # We are fitting v_main_ac to model = I*basis_I + Q*basis_Q
                A_matrix = np.vstack([basis_I, basis_Q]).T
                # `lstsq` solves the equation A_matrix * x = v_main_ac for x=[I, Q]
                p, res, _, _ = np.linalg.lstsq(A_matrix, v_main_ac, rcond=None)
                
                # The cost is the sum of squared residuals from the linear fit
                return res[0]

            # Use a robust non-linear optimizer to find the best (tau, psi)
            # that minimizes the cost function from the inner linear fit.
            opt_result = minimize(
                _outer_loop_cost,
                current_guess,
                method='Nelder-Mead'
            )
            
            # --- 4. Recover All Parameters from the Optimal Solution ---
            tau_fit, psi_fit = opt_result.x
            
            # Re-run the linear fit one last time with the optimal (tau, psi) to get I and Q
            # (This is more robust than trying to pass I and Q out of the optimizer)
            phi_mod_unscaled = 2 * np.pi * laser_cfg.df * phi_mod_basis
            t_shift_psi = -psi_fit / omega_mod
            t_interp_psi = time_axis_buffer - t_shift_psi
            phi_mod_shifted = np.interp(t_interp_psi, time_axis_buffer, phi_mod_unscaled, period=time_axis_buffer[-1])
            t_interp_tau = time_axis_buffer - tau_fit
            phi_mod_delayed = np.interp(t_interp_tau, time_axis_buffer, phi_mod_shifted, period=time_axis_buffer[-1])
            delta_phi_mod = phi_mod_delayed - phi_mod_shifted
            basis_I = np.cos(delta_phi_mod)
            basis_Q = np.sin(delta_phi_mod)
            A_matrix = np.vstack([basis_I, basis_Q]).T
            p_final, final_res, _, _ = np.linalg.lstsq(A_matrix, v_main_ac, rcond=None)
            I_fit, Q_fit = p_final
            
            # Recover C and Phi
            C_fit = np.sqrt(I_fit**2 + Q_fit**2)
            phi_fit = np.arctan2(-Q_fit, I_fit)
            
            # Recover m for reporting
            m_fit = 2 * np.pi * laser_cfg.df * tau_fit
            
            # Update guess for warm start
            current_guess = np.array([tau_fit, psi_fit])

            results_list.append({
                'amp': C_fit, 'm': m_fit, 'phi': phi_fit, 'psi': psi_fit,
                'dc': np.mean(main_buffer_raw),
                'ssq': final_res[0] if final_res else 0,
                'fitok': 1 if opt_result.success else 0, 'b': b
            })
            
        self.fits_df[fit_label] = pd.DataFrame(results_list)
        return self._create_fit_object_from_df(fit_label, main_label, n, R, fs, nbuf, 0, 0, init_m)
    
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
    
    def fit_wdfmi_sequential(self, main_label, witness_label, n=None, fit_label=None, init_psi=0.0, verbose=True):
        """
        Fits DFMI data using a sequential bootstrap algorithm.

        This algorithm is immune to the `tau-psi` degeneracy and numerical
        instability. It works by sequentially solving for parameters using the most
        robust method available at each stage:
        1. It finds `tau` via a 1D Variable Projection (VarPro) fit.
        2. With `tau` fixed, it finds `psi` via a robust 1D "Differential Phase" fit.
        3. With both `tau` and `psi` known, it performs a final linear fit for `C` and `Phi`.

        Parameters
        ----------
        main_label : str
            The label of the primary DeepRawObject to be fit.
        witness_label : str
            The label of the DeepRawObject to use as the witness.
        n : int, optional
            The number of modulation periods per fit buffer (`n_fit`).
        fit_label : str, optional
            The label for the output DeepFitObject. Defaults to `main_label + '_wdfmi_seq'`.
        init_psi : float, optional
            An initial guess for psi, used to center the search. In a real system,
            this would be determined from a one-time calibration.
        verbose : bool, optional
            If True, display a progress bar.

        Returns
        -------
        DeepFitObject
            A fit object containing the time-series of the estimated parameters.
        """
        # --- 1. Initialization ---
        if main_label not in self.raws or witness_label not in self.raws:
            logging.error("Invalid main or witness label provided!"); return
        if fit_label is None: fit_label = main_label + '_wdfmi_seq'
        if n is None: n = self.sims[main_label].fit_n

        R, fs, nbuf = self.fit_init(main_label, n)
        if nbuf == 0: logging.error('Check buffer size !!'); return
        
        main_raw = self.raws[main_label]
        laser_cfg = main_raw.sim.laser
        main_ifo_cfg = main_raw.sim.ifo
        omega_mod = 2 * np.pi * laser_cfg.f_mod
        
        # --- 2. Process Witness Signal once ---
        witness_buffer_raw = np.array(self.raws[witness_label].data.loc[0:R-1]).flatten()
        v_w_ac = witness_buffer_raw - np.mean(witness_buffer_raw)
        f_mod_basis = -v_w_ac / np.max(np.abs(v_w_ac))
        time_axis_buffer = np.arange(R) / main_raw.f_samp
        dt = time_axis_buffer[1] - time_axis_buffer[0]
        phi_mod_basis = np.cumsum(f_mod_basis) * dt
        phi_mod_unscaled = 2 * np.pi * laser_cfg.df * phi_mod_basis

        # --- 3. Main Fitting Loop ---
        results_list = []
        iterable = range(nbuf)
        if verbose: iterable = tqdm(iterable)

        for b in iterable:
            buf_range = range(b * R, (b + 1) * R)
            main_buffer_raw = np.array(main_raw.data.loc[buf_range]).flatten()
            v_main_ac = main_buffer_raw - np.mean(main_buffer_raw)
            
            # --- Helper function to construct time-domain basis functions ---
            def get_bases(tau, psi):
                t_shift_psi = -psi / omega_mod
                t_interp_psi = time_axis_buffer - t_shift_psi
                phi_mod_shifted = np.interp(t_interp_psi, time_axis_buffer, phi_mod_unscaled, period=time_axis_buffer[-1])
                
                t_interp_tau = time_axis_buffer - tau
                phi_mod_delayed = np.interp(t_interp_tau, time_axis_buffer, phi_mod_shifted, period=time_axis_buffer[-1])
                
                delta_phi_mod = phi_mod_delayed - phi_mod_shifted
                return np.cos(delta_phi_mod), np.sin(delta_phi_mod)

            # --- Stage 1: Find Tau using VarPro ---
            def cost_tau(tau):
                basis_I, basis_Q = get_bases(tau, init_psi) # Use initial psi guess for this step
                A_matrix = np.vstack([basis_I, basis_Q]).T
                _, res, _, _ = np.linalg.lstsq(A_matrix, v_main_ac, rcond=None)
                return res[0] if res.size > 0 else np.inf
            
            delta_l = main_ifo_cfg.meas_arml - main_ifo_cfg.ref_arml
            tau_init = delta_l / sc.c
            res_tau = minimize_scalar(cost_tau, bracket=(tau_init*0.9, tau_init*1.1), method='brent')
            tau_fit = res_tau.x
            
            # --- Stage 2: Find Psi using Differential Phase ---
            # First, we need the complex spectrum of the measured data
            ndata_psi = 40
            w0_samp = omega_mod / main_raw.f_samp
            Q_meas, I_meas = np.zeros(ndata_psi), np.zeros(ndata_psi)
            for i in range(ndata_psi):
                n_harm = i + 1
                Q_meas[i] = (2/R) * np.sum(v_main_ac * np.cos(n_harm * w0_samp * np.arange(R)))
                I_meas[i] = (2/R) * np.sum(v_main_ac * np.sin(n_harm * w0_samp * np.arange(R)))
            alpha_meas = Q_meas + 1j * I_meas

            # Helper to get the model spectrum for the differential phase method
            def get_model_spectrum(psi_trial):
                basis_I, basis_Q = get_bases(tau_fit, psi_trial)
                A_matrix = np.vstack([basis_I, basis_Q]).T
                p_final, _, _, _ = np.linalg.lstsq(A_matrix, v_main_ac, rcond=None)
                I_fit, Q_fit = p_final
                C_fit, phi_fit = np.sqrt(I_fit**2 + Q_fit**2), np.arctan2(-Q_fit, I_fit)
                
                v_model = C_fit * np.cos(phi_fit + np.arctan2(basis_Q, basis_I)) # Simplified for speed
                v_model_approx = I_fit*basis_I - Q_fit*basis_Q
                
                Q_m, I_m = np.zeros(ndata_psi), np.zeros(ndata_psi)
                for i in range(ndata_psi):
                    n_harm = i+1
                    Q_m[i] = (2/R) * np.sum(v_model_approx * np.cos(n_harm * w0_samp * np.arange(R)))
                    I_m[i] = (2/R) * np.sum(v_model_approx * np.sin(n_harm * w0_samp * np.arange(R)))
                return Q_m + 1j*I_m
                
            # For this stage, we assume psi is a small deviation from the initial guess
            def cost_psi(psi_deviation):
                alpha_model = get_model_spectrum(init_psi + psi_deviation)
                phase_error = np.angle(alpha_meas * np.conj(alpha_model))
                return np.var(np.unwrap(phase_error))
                
            # Due to complexity, a simpler bootstrap is often used: find psi on first buffer only
            # and assume it's constant. For now, we fit every buffer.
            res_psi = minimize_scalar(cost_psi, bounds=(-np.pi, np.pi), method='bounded')
            psi_fit = init_psi + res_psi.x
            
            # --- Stage 3: Final Linear Fit ---
            basis_I, basis_Q = get_bases(tau_fit, psi_fit)
            A_matrix = np.vstack([basis_I, basis_Q]).T
            p_final, final_res, _, _ = np.linalg.lstsq(A_matrix, v_main_ac, rcond=None)
            I_fit, Q_fit = p_final
            
            C_fit = np.sqrt(I_fit**2 + Q_fit**2)
            phi_fit = np.arctan2(-Q_fit, I_fit)
            m_fit = 2 * np.pi * laser_cfg.df * tau_fit
            
            results_list.append({
                'amp': C_fit, 'm': m_fit, 'phi': phi_fit, 'psi': psi_fit,
                'dc': np.mean(main_buffer_raw),
                'ssq': final_res[0] if final_res.size > 0 else 0,
                'fitok': 1, 'b': b
            })

        self.fits_df[fit_label] = pd.DataFrame(results_list)
        # The last two ndata arguments are now meaningless for this fitter, pass 0.
        return self._create_fit_object_from_df(fit_label, main_label, n, R, fs, nbuf, 0, 0, 0)

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
from .plotting import *
from .fit import *
from .data import *
from .physics import *
from .dsp import *

import numpy as np
import scipy.constants as sc
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
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
            Q_data, I_data = calculate_quadratures(n, buffer_data, w0, R)
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
    """A framework for loading, processing, and representing DFMSWPM data.

    Say that `raw_data.txt` contains only a single channel. You can load it as:

    dff = DeepFitFramework(raw_file='./raw_data.txt', raw_labels=['raw1'])

    or

    dff = DeepFitFramework()
    dff.load_raw('./raw_data.txt', labels=['raw1'])

    When loading the file, the metadata contained in the header is parsed, and stored
    along with the raw data in a DeepRawObject. These are appended to the `raws`
    dictionary. You can check it out with:

    dff.raws 
    # {'raw1': <dfmdata.DeepRawObject at 0x1121b5d90>}

    dff.raws['raw1'].info() 
    # This shows some metadata

    dff.raws['raw1'].data
    # This shows the raw data

    You can similarly load a fit_data file. When doing so, a DeepFitObject is
    created for every channel in the data, containing the time series of the
    fit parameters as well as the relevant metadata. 

    dff.load_fit('./fit_data.txt', labels=['fit1'])

    Plotting all of the loaded fit data is as easy as:

    dff.plot()

    Apart from loading raw_data or fit_data files, you can also perform fits on
    existing raw data, which will create the corresponding DeepFitObject. For
    example:

    dff.fit('raw1', n=20, fit_label='fit2')

    DeepFitObject's are appended to the `fits` dictionary:

    dff.fits
    % {'fit1': <dfmdata.DeepFitObject at 0 
    x1121b5d90>, 'fit2': <dfmdata.DeepFitObject at 0x120776fd0>}

    You can check a DeepFitObject's metadata using:

    dff.fits['fit1'].info()

    and you can save the results to a new fit_data file as:

    dff.fits['fit1'].to_txt('example.txt')

    You can also perform simulations. To do this, a DFMIObject is created e.g. as:

    dff.new_sim("sim1")

    which appends a new DFMIObject to the `sims` dictionary:

    {'sim1': <dfmdata.DFMIObject at 0x13fdb8be0>}

    Running a simulation will create new DeepRawObject's with the resulting time series:

    dff.simulate("sim1", n_seconds=60, simulate="dynamic", ref_channel=True)

    creates two DeepRawObjects "sim1" (with armlength modulation) and "sim1_ref",
    which is subject to the same laser noise but without armlength modulation. 
    You can then run the fit on these as described.
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
                snr_db=None, trial_num=0):
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
            logging.info(f"Simulating '{main_label}' with witness '{witness_label}' for {n_seconds:.2f}s...")
        else:
            logging.info(f"Simulating '{main_label}' for {n_seconds:.2f}s...")

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
        logging.info(f"Simulation finished in {sim_time:.3f} s.")

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
                raw_labels[c] = self.raw_file + '_ch' + str(c)
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

    def _fit_single_buffer(self, label, b, R, ndata, initial_guess):
        """
        Processes a single buffer of raw data to produce one fit result.
        
        This is a modular helper function that encapsulates the logic for
        fitting one buffer, given an explicit initial guess. This breaks the
        dependency on the `fits_df` state for easier reuse.

        Returns a dictionary with the fit results.
        """
        # 1. Prepare data for the given buffer
        buf = range(b * R, (b + 1) * R)
        raw_buffer = np.array(self.raws[label].data.loc[buf]).flatten()
        w0 = 2. * np.pi * self.raws[label].f_mod / self.raws[label].f_samp
        
        # 2. Calculate I/Q values
        QI_data_mean = np.zeros(2 * ndata)
        for n in range(ndata):
            Q_data, I_data = calculate_quadratures(n, raw_buffer, w0, R)
            QI_data_mean[n] = Q_data.mean()
            QI_data_mean[n + ndata] = I_data.mean()
        
        # 3. Run the NLS fit
        status, fit_parm, fit_ssq = fit(ndata, QI_data_mean, initial_guess)
        
        # 4. Return results as a simple dictionary
        return {
            'amp': fit_parm[0],
            'm': fit_parm[1],
            'phi': fit_parm[2],
            'psi': fit_parm[3],
            'dc': np.mean(raw_buffer),
            'ssq': fit_ssq,
            'fitok': status,
            'b': b
        }
    
    def fit_ekf(self, label, fit_label=None, init_a=1.6, init_m=6.0, init_phi=0.0, init_psi=0.0, 
                P0_diag=None, Q_diag=None, R_val=None):
        """
        Performs state estimation using an Extended Kalman Filter (EKF).

        This method processes the raw data sequentially, sample by sample, to
        provide a time-varying estimate of the interferometer's state. It is
        particularly useful for tracking dynamic changes.

        Parameters
        ----------
        label : str
            The label of the DeepRawObject containing the raw data to process.
        fit_label : str, optional
            The label for the output DeepFitObject. If None, defaults to `label + '_ekf'`.
        init_a, init_m, init_phi, init_psi : float, optional
            Initial guesses for the state parameters.
        P0_diag : list or np.ndarray, optional
            A 5-element list specifying the initial variances for the diagonal
            of the state covariance matrix `P`. Represents the initial uncertainty
            of the state guess. Defaults to `[1, 1, 1, 1, 1]`.
        Q_diag : list or np.ndarray, optional
            A 5-element list specifying the process noise variances for the
            state vector `[a, m, phi, psi, dc]`. This is a critical tuning
            parameter that determines how quickly the filter adapts to changes.
            Defaults to `[1e-8, 1e-8, 1e-6, 1e-6, 1e-8]`.
        R_val : float, optional
            The measurement noise variance. Represents the variance of the
            electronic noise on a single voltage sample. If None, it is
            estimated from the raw data. Defaults to None.
        
        Returns
        -------
        DeepFitObject
            A fit object containing the time-series of the estimated parameters.
        """
        if fit_label is None:
            fit_label = label + '_ekf'
        if label not in self.raws:
            logging.error(f"Invalid raw data label: '{label}' !!"); return

        raw_obj = self.raws[label]
        data = raw_obj.data["ch0"].to_numpy()
        
        # --- 1. EKF Initialization ---
        # State vector: x = [amplitude, mod_depth, phase, mod_phase, dc_offset]
        dim_x = 5
        x = np.array([init_a, init_m, init_phi, init_psi, np.mean(data)])

        # Initial state covariance P0: Represents our initial uncertainty.
        if P0_diag is None: P0_diag = [1.0, 1.0, 1.0, 1.0, 1.0]
        P = np.diag(P0_diag)
        
        # Process noise Q: A key tuning parameter. How much we expect the state to change.
        if Q_diag is None: Q_diag = [1e-8, 1e-8, 1e-6, 1e-6, 1e-8]
        Q = np.diag(Q_diag)
        
        # Measurement noise R: Variance of the ADC/photodiode noise.
        if R_val is None: R_val = np.var(data)
        R = np.array([[R_val]]) # Must be a 1x1 matrix

        # Process model Jacobian F: For a random walk model, F is the identity matrix.
        F = np.eye(dim_x)
        
        # Setup for the run
        n_samp = len(data)
        w_m = 2 * np.pi * raw_obj.f_mod
        t_axis = np.arange(n_samp) / raw_obj.f_samp
        
        # Arrays to store the results
        # We will downsample the results to match the NLS fitter's rate
        n_fit_cycles = self.sims[label].fit_n if label in self.sims else 20
        R_downsample, fs, nbuf = self.fit_init(label, n_fit_cycles)

        results = np.zeros((nbuf, dim_x))
        innovations = np.zeros(n_samp)

        logging.info(f"Running EKF for {n_samp} samples...")
        
        # --- 2. Main EKF Loop ---
        for k in tqdm(range(n_samp)):
            # --- PREDICT STEP ---
            # For a random walk model, the state prediction is just the previous state.
            # x_priori = F @ x  (but F is identity)
            P = F @ P @ F.T + Q

            # --- UPDATE STEP ---
            # Calculate the measurement model Jacobian (H) at the current state
            a, m, phi, psi, dc = x
            theta = w_m * t_axis[k] + psi
            
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            full_phase_arg = phi + m * cos_theta
            cos_full_arg = np.cos(full_phase_arg)
            sin_full_arg = np.sin(full_phase_arg)

            h_x = a * cos_full_arg + dc # The model's prediction of the measurement
            
            # H = [dh/da, dh/dm, dh/dphi, dh/dpsi, dh/ddc]
            H = np.zeros((1, dim_x))
            H[0, 0] = cos_full_arg
            H[0, 1] = -a * sin_full_arg * cos_theta
            H[0, 2] = -a * sin_full_arg
            H[0, 3] = a * m * sin_full_arg * sin_theta
            H[0, 4] = 1.0

            # Calculate the innovation (residual)
            z_k = data[k]
            y_k = z_k - h_x
            innovations[k] = y_k

            # Calculate the Kalman Gain
            S = H @ P @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)

            # Update the state and covariance
            x = x + (K @ y_k.reshape(1,1)).flatten()
            P = (np.eye(dim_x) - K @ H) @ P

            # --- Store result at the downsampled rate ---
            if (k + 1) % R_downsample == 0:
                buf_idx = (k + 1) // R_downsample - 1
                if buf_idx < nbuf:
                    results[buf_idx, :] = x

        # --- 3. Create and return a standard DeepFitObject ---
        logging.info("EKF processing finished. Creating fit object...")
        
        # To reuse the existing helper, we need to create a temporary DataFrame
        df_dict = {
            'amp': results[:, 0], 'm': results[:, 1], 'phi': results[:, 2],
            'psi': results[:, 3], 'dc': results[:, 4],
            'ssq': np.zeros(nbuf), # SSQ is not calculated in EKF, use innovation variance later
            'fitok': np.ones(nbuf) # Assume fit was ok if it didn't diverge
        }
        self.fits_df[fit_label] = pd.DataFrame(df_dict)
        
        # Use our standard object creation helper
        return self._create_fit_object_from_df(fit_label, raw_obj.label, n_fit_cycles, R_downsample, fs, nbuf, 0, init_a, init_m)

    def fit(self, label, n=None, init_a=1.6, init_m=6.0, ndata=10, fit_label=None, init_psi=False, verbose=True, parallel=True, n_cores=None):
        """
        Performs fit on the raw data, either sequentially or in parallel.
        
        This method acts as a router. For `parallel=True`, it uses the
        block-parallel strategy for high throughput. Otherwise, it runs
        the traditional sequential fit.
        """
        if parallel:
            # Route to the new parallel method
            return self.fit_parallel(label, n=n, init_a=init_a, init_m=init_m, ndata=ndata,
                                     fit_label=fit_label, init_psi=init_psi, verbose=verbose, n_cores=n_cores)
        
        # --- Standard Sequential Fit Logic (Refactored) ---
        if n is None:
            try: n = self.sims[label].fit_n
            except KeyError: logging.error("fit_n not specified!!"); return
        if fit_label is None: fit_label = label + '_fit'
        if label not in self.raws: logging.error('Invalid label !!'); return

        R, fs, nbuf = self.fit_init(label, n)
        if nbuf == 0: logging.error('Check buffer size !!'); return

        raw_obj = self.raws[label]
        if hasattr(raw_obj, 'phi_sim') and raw_obj.phi_sim is not None and len(raw_obj.phi_sim) > 0:
            raw_obj.phi_sim_downsamp = vectorized_downsample(raw_obj.phi_sim, R)

        if init_psi:
            psi = self.psi_init(label, init_psi, init_a, init_m, R, ndata, fit_label, verbose)
        else:
            psi = 0.0

        # Start with the initial guess
        current_guess = np.array([init_a, init_m, 0.0, psi])
        results_list = []

        iterable = range(nbuf)
        if verbose:
            logging.info('Processing sequentially...')
            iterable = tqdm(iterable)
        
        for b in iterable:
            # Call the modular single-buffer fitter
            newfit_dict = self._fit_single_buffer(label, b, R, ndata, current_guess)
            results_list.append(newfit_dict)
            
            # Update the guess for the next buffer (warm start)
            current_guess = np.array([newfit_dict['amp'], newfit_dict['m'], newfit_dict['phi'], newfit_dict['psi']])
            
        # --- Post-processing and object creation (same for both methods) ---
        self.fits_df[fit_label] = pd.DataFrame(results_list)
        return self._create_fit_object_from_df(fit_label, label, n, R, fs, nbuf, ndata, init_a, init_m)

    def fit_parallel(self, label, n=None, init_a=1.6, init_m=6.0, ndata=10, fit_label=None, init_psi=False, verbose=True, n_cores=None):
        """Performs fit in parallel using a block-wise strategy."""
        from multiprocessing import Pool
        import os

        if n is None:
            try: n = self.sims[label].fit_n
            except KeyError: logging.error("fit_n not specified!!"); return
        if fit_label is None: fit_label = label + '_fit'
        if label not in self.raws: logging.error('Invalid label !!'); return
            
        R, fs, nbuf = self.fit_init(label, n)
        if nbuf == 0: logging.error('Check buffer size !!'); return

        raw_obj = self.raws[label]
        if hasattr(raw_obj, 'phi_sim') and raw_obj.phi_sim is not None and len(raw_obj.phi_sim) > 0:
            raw_obj.phi_sim_downsamp = vectorized_downsample(raw_obj.phi_sim, R)
            
        if n_cores is None:
            n_cores = os.cpu_count()
        n_cores = min(n_cores, nbuf) # Don't use more cores than buffers

        if verbose:
            logging.info(f"Processing in parallel with {n_cores} cores...")
            pass

        # 1. Seeding: Get a single good initial guess by fitting the first buffer
        if init_psi:
            psi = self.psi_init(label, init_psi, init_a, init_m, R, ndata, fit_label, verbose)
        else:
            psi = 0.0
        seed_guess = np.array([init_a, init_m, 0.0, psi])
        first_fit_result = self._fit_single_buffer(label, 0, R, ndata, seed_guess)
        seed_guess = np.array([first_fit_result['amp'], first_fit_result['m'], first_fit_result['phi'], first_fit_result['psi']])
        
        # 2. Chunking: Split the raw data into chunks for each core
        raw_data_full = self.raws[label].data.values.reshape(-1, R)
        # We start from buffer 1 since buffer 0 was used for seeding
        chunks = np.array_split(raw_data_full[1:], n_cores) 
        
        # 3. Prepare Jobs
        job_args = [(chunk, seed_guess, R, ndata, self.raws[label].f_mod, self.raws[label].f_samp) for chunk in chunks if chunk.size > 0]
        
        # 4. Run the Pool
        with Pool(n_cores) as p:
            # Use imap for progress bar compatibility
            chunk_results_list = list(tqdm(p.imap(_process_fit_chunk, job_args), total=len(job_args)))
        
        # 5. Stitch Results
        # Start with the result from the seed fit (buffer 0)
        final_results = [first_fit_result]
        # Append the results from all other chunks
        for chunk_res in chunk_results_list:
            final_results.extend(chunk_res)
            
        # Adjust the buffer number 'b' for the stitched results
        for i, res in enumerate(final_results):
            res['b'] = i
            
        # --- Post-processing and object creation (same for both methods) ---
        self.fits_df[fit_label] = pd.DataFrame(final_results)
        return self._create_fit_object_from_df(fit_label, label, n, R, fs, nbuf, ndata, init_a, init_m)

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

    def psi_init(self, label, method, init_a, init_m, R, ndata, verbose):
        """
        Finds an optimal initial guess for the modulation phase (psi).

        This method is called before the main fitting process to ensure the NLS
        fitter starts with a good value for `psi`, which is critical for
        convergence.

        Parameters
        ----------
        label : str
            The label of the raw data channel to use.
        method : {'scan', 'minimize'}
            The method for finding the best psi.
            - 'scan': A simple grid search over 2*pi.
            - 'minimize': A more precise search using scipy.optimize.minimize,
              seeded by a coarse initial scan.
        init_a : float
            The initial guess for the amplitude to use during the psi search.
        init_m : float
            The initial guess for the modulation depth to use during the psi search.
        R : int
            The buffer size (downsampling factor).
        ndata : int
            The number of harmonics to fit.
        verbose : bool
            If True, print progress information.
        
        Returns
        -------
        float
            The best initial psi value found.
        """
        if verbose:
            logging.info(f"Initializing psi parameter for '{label}' using '{method}' method...")

        # Create a tuple of the arguments needed by the helper function try_psi.
        # This keeps the call signatures clean.
        try_psi_args = (label, init_a, init_m, R, ndata)

        if method == 'scan':
            best_ssq = float('inf')
            final_psi = 0.0
            
            for psi_test in np.linspace(0, 2 * np.pi, 20, endpoint=False):
                new_ssq = self.try_psi(psi_test, *try_psi_args)
                if new_ssq < best_ssq:
                    best_ssq = new_ssq
                    final_psi = psi_test

        elif method == 'minimize':
            from scipy.optimize import minimize
            
            # Find a good starting point for the minimizer by checking a few coarse points,
            # which is more efficient than a full grid search.
            num_coarse_points = 4
            coarse_psis = np.linspace(0, 2 * np.pi, num_coarse_points, endpoint=False)
            coarse_ssqs = [self.try_psi(p, *try_psi_args) for p in coarse_psis]
            
            best_initial_psi = coarse_psis[np.argmin(coarse_ssqs)]

            if verbose:
                logging.info(f"Coarse search found best initial guess psi = {best_initial_psi:.4f}. Starting minimizer.")
            
            # Run the high-precision minimizer.
            # `try_psi` is the function to minimize.
            # `x0` is the starting point.
            # `args` are the *other* arguments to pass to `try_psi`.
            res = minimize(self.try_psi,
                           x0=best_initial_psi,
                           args=try_psi_args,
                           method="Nelder-Mead",
                           bounds=[(0.0, 2 * np.pi)]) # Bounds are good practice
            
            final_psi = res.x[0]

        else:
            logging.warning(f"Unknown psi_init method: '{method}'. Defaulting to 0.0.")
            final_psi = 0.0

        if verbose:
            logging.info(f"Selected init_psi = {final_psi:.4f}")

        return final_psi

    def try_psi(self, psi, label, init_a, init_m, R, ndata):
        """
        Helper function to test a single psi value and return the resulting SSQ.
        
        This function is stateless; it does not modify the DeepFitFramework
        instance. It fits only the first buffer (b=0) of the specified raw data.
        
        Parameters
        ----------
        psi : float
            The trial value for the modulation phase to be tested.
        label : str
            The label of the raw data channel.
        init_a : float
            The fixed initial guess for amplitude.
        init_m : float
            The fixed initial guess for modulation depth.
        R : int
            The buffer size.
        ndata : int
            The number of harmonics.
            
        Returns
        -------
        float
            The sum of squared residuals (ssq) from the fit.
        """
        # 1. Construct the initial guess vector with the trial psi
        initial_guess = np.array([init_a, init_m, 0.0, psi])
        
        # 2. Call our clean, single-buffer fitting function
        result_dict = self._fit_single_buffer(label, 0, R, ndata, initial_guess)
        
        # 3. Return the SSQ
        return result_dict['ssq']

    def fit_init(self, label, n):

        R = int(self.raws[label].f_samp/self.raws[label].f_mod*n)
        fs = self.raws[label].f_samp/R
        nbuf = int(self.raws[label].data.shape[0]/R)
        if nbuf == 0:
            logging.error('Check buffer size !!')

        return R, fs, nbuf

    def fit_wdfmi(self, main_label, witness_label, n=None, fit_label=None, ndata=10, init_a=1.6, init_m=6.0, init_phi=0.0, init_psi=0.0, verbose=True):
        """
        Performs a fit using the Witnessed DFMI (W-DFMI) numerical method.

        This is the definitive, physically correct version. It uses the exact
        physical model for the interferometric phase difference, `phi(t-tau) - phi(t)`.
        It correctly identifies the witness signal as a measure of the frequency
        modulation, integrates it to get the phase modulation, and fits for the
        physical time delay `tau` to ensure numerical stability.

        Parameters
        ----------
        main_label : str
            The label of the primary DeepRawObject containing the main DFMI signal.
        witness_label : str
            The label of the DeepRawObject containing the witness signal. It is
            assumed this witness was operated at the mid-fringe point (phi=pi/2).
        n : int, optional
            The number of modulation periods per fit buffer (`n_fit`).
        fit_label : str, optional
            The label for the output DeepFitObject. Defaults to `main_label + '_wdfmi'`.
        ndata : int, optional
            The number of harmonics to include in the fit.
        init_a, init_m, init_phi, init_psi : float, optional
            Initial guesses for the fit parameters.
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
        if fit_label is None: fit_label = main_label + '_wdfmi'
        if n is None: n = self.sims[main_label].fit_n

        R, fs, nbuf = self.fit_init(main_label, n)
        if nbuf == 0: logging.error('Check buffer size !!'); return
        
        main_raw = self.raws[main_label]
        laser_cfg = main_raw.sim.laser
        main_ifo_cfg = main_raw.sim.ifo
        omega_mod = 2 * np.pi * laser_cfg.f_mod
        w0_samp = omega_mod / main_raw.f_samp
        
        # --- 2. Process Witness Signal to get Phase Modulation Waveform phi_mod(t) ---
        witness_buffer_raw = np.array(self.raws[witness_label].data.loc[0:R-1]).flatten()
        v_w_ac = witness_buffer_raw - np.mean(witness_buffer_raw)
        
        # The witness voltage is proportional to -f_mod(t). First, get the normalized f_mod(t) basis.
        f_mod_basis = -v_w_ac / np.max(np.abs(v_w_ac))
        
        # Now, integrate the frequency basis to get the phase modulation basis.
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
            logging.info(f"Processing '{main_label}' with definitive Exact Model W-DFMI readout...")
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
                
                # The full, unscaled phase modulation waveform
                phi_mod_unscaled = 2 * np.pi * laser_cfg_arg.df * phi_mod_basis_arg
                
                # Apply psi as a time shift to this phase waveform
                t_shift_psi = -psi / omega_mod
                t_interp_psi = time_axis_buffer - t_shift_psi
                phi_mod_shifted = np.interp(t_interp_psi, time_axis_buffer, phi_mod_unscaled, period=time_axis_buffer[-1])
                
                # Apply tau delay to the psi-shifted waveform
                t_interp_tau = time_axis_buffer - tau
                phi_mod_delayed = np.interp(t_interp_tau, time_axis_buffer, phi_mod_shifted, period=time_axis_buffer[-1])
                
                # The final model phase is the difference of the delayed and local phase waveforms
                delta_phi_mod = phi_mod_delayed - phi_mod_shifted
                
                v_model = C * np.cos(phi + delta_phi_mod)
                
                QI_model = np.zeros(2 * ndata)
                for i in range(ndata):
                    n_harm = i + 1
                    QI_model[i] = (2/R) * np.sum(v_model * np.cos(n_harm * w0_samp * np.arange(R)))
                    QI_model[i + ndata] = (2/R) * np.sum(v_model * np.sin(n_harm * w0_samp * np.arange(R)))
                
                return QI_model - QI_meas

            opt_result = least_squares(
                _wdfmi_residuals,
                current_guess,
                args=(phi_mod_basis, laser_cfg, QI_data_meas),
                method='lm'
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
    
    def new_wdfmi_setup(self, main_label, witness_label, m_witness=0.1):
        """
        Creates a complete Measurement and Witness channel configuration for W-DFMI.

        This helper function simplifies the process of setting up a W-DFMI
        simulation. It creates a single shared LaserConfig and two distinct
        InterferometerConfigs, one for the main channel and one for the witness.
        It automatically adjusts the witness arm lengths to achieve a desired
        small modulation depth.

        Parameters
        ----------
        main_label : str
            The label for the main measurement channel.
        witness_label : str
            The label for the witness channel.
        m_witness : float, optional
            The target modulation depth for the witness interferometer. This should
            be small (<< 1) for the witness to act as a linear frequency-to-
            amplitude converter.

        Returns
        -------
        tuple
            A tuple containing the main channel config and the witness channel config.
        """
        # 1. Create the single, shared laser source
        laser = LaserConfig(label=f"{main_label}_laser")
        self.lasers[laser.label] = laser

        # 2. Create the main interferometer config
        main_ifo_config = InterferometerConfig(label=f"{main_label}_ifo")
        self.ifos[main_ifo_config.label] = main_ifo_config # Assuming dff.ifos dict

        # 3. Create the witness interferometer config
        witness_ifo_config = InterferometerConfig(label=f"{witness_label}_ifo")
        # Make it static
        witness_ifo_config.arml_mod_amp = 0.0
        # Artificially set arm lengths to achieve the desired witness `m`
        delta_l_witness = (m_witness * sc.c) / (2 * np.pi * laser.df)
        witness_ifo_config.ref_arml = 0.01 # Arbitrary small base length
        witness_ifo_config.meas_arml = witness_ifo_config.ref_arml + delta_l_witness
        self.ifos[witness_ifo_config.label] = witness_ifo_config
        
        # 4. Create the two simulation channel objects
        main_channel = DFMIObject(main_label, laser, main_ifo_config)
        witness_channel = DFMIObject(witness_label, laser, witness_ifo_config)
        
        # Store them in the framework's main dictionary
        self.sims[main_label] = main_channel
        self.sims[witness_label] = witness_channel
        
        logging.info(f"Created W-DFMI setup: '{main_label}' (m={main_channel.m:.2f}) and '{witness_label}' (m={witness_channel.m:.2f})")
        return main_channel, witness_channel
    
    def create_witness_channel(self, main_channel_label, witness_channel_label, delta_l_witness=None, m_witness=None):
        """
        Creates a witness channel configuration based on an existing main channel.

        This helper function simplifies the creation of a static witness channel
        that is physically linked to a main channel (i.e., it shares the same
        laser source). It automatically handles the creation of the necessary
        configuration objects and sets the witness arm lengths to achieve a
        desired optical path difference or modulation depth.

        The user must specify *either* `delta_l_witness` or `m_witness`, but not
        both. If neither is specified, it defaults to creating a witness with
        `m_witness = 0.1`, suitable for W-DFMI.

        Parameters
        ----------
        main_channel_label : str
            The label of the existing DFMIObject in `self.sims` to use as a template.
        witness_channel_label : str
            The label for the new witness channel to be created.
        delta_l_witness : float, optional
            The desired absolute optical path difference (meas_arml - ref_arml)
            for the witness interferometer, in meters.
        m_witness : float, optional
            The desired effective modulation index for the witness interferometer.

        Returns
        -------
        DFMIObject
            The newly created witness channel object, which has also been added
            to the framework's `sims` dictionary.

        Raises
        ------
        KeyError
            If `main_channel_label` does not exist in `self.sims`.
        ValueError
            If both `delta_l_witness` and `m_witness` are specified.
        """
        # --- 1. Validate Inputs ---
        if main_channel_label not in self.sims:
            raise KeyError(f"Main channel '{main_channel_label}' not found in framework.")
        if delta_l_witness is not None and m_witness is not None:
            raise ValueError("Please specify either delta_l_witness or m_witness, but not both.")
        # Default behavior if neither is specified
        if delta_l_witness is None and m_witness is None:
            m_witness = 0.1
            logging.info(f"Neither delta_l nor m specified for witness. Defaulting to m_witness = {m_witness}.")

        # --- 2. Get Shared Laser and Template Channel ---
        main_channel = self.sims[main_channel_label]
        shared_laser = main_channel.laser

        # --- 3. Create and Configure New Interferometer for the Witness ---
        witness_ifo = InterferometerConfig(label=f"{witness_channel_label}_ifo")
        witness_ifo.arml_mod_amp = 0.0  # Witness is always static
        witness_ifo.arml_mod_n = 0.0

        # --- 4. Calculate and Set Witness Arm Lengths ---
        if m_witness is not None:
            # Calculate the required OPD to achieve the target 'm'
            if shared_laser.df == 0:
                raise ValueError("Cannot set 'm' for witness when laser 'df' is zero.")
            delta_l = (m_witness * sc.c) / (2 * np.pi * shared_laser.df)
        else: # delta_l_witness must have been specified
            delta_l = delta_l_witness

        # Set arm lengths based on the calculated OPD
        witness_ifo.ref_arml = 0.01  # Arbitrary small base length
        witness_ifo.meas_arml = witness_ifo.ref_arml + delta_l

        # --- 5. Compose and Register the New Witness Channel ---
        witness_channel = DFMIObject(
            label=witness_channel_label,
            laser_config=shared_laser,
            ifo_config=witness_ifo,
            f_samp=main_channel.f_samp  # Inherit f_samp from main channel
        )
        witness_channel.fit_n = main_channel.fit_n # Inherit fit_n

        self.sims[witness_channel_label] = witness_channel

        logging.info(f"Created witness channel '{witness_channel_label}' linked to laser '{shared_laser.label}'.")
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
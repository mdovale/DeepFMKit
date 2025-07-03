from .plotting import *
from .fit import *
from .data import *

import numpy as np
import scipy.constants as sc
import pandas as pd
from tqdm import tqdm
import pyplnoise
import time
import copy
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

    def _generate_noise_arrays(self, sim_config, time_axis, trial_num=0):
        """
        Helper function to generate all required noise time-series arrays.

        This version is optimized to minimize expensive calls to the noise
        generation library. It generates one "basis" noise for each unique
        spectral shape (alpha value) and scales it to produce the final
        noise arrays. This also fixes a bug where all noise sources were
        spuriously correlated due to a fixed random seed.
        """
        num_samples = len(time_axis)
        fs = sim_config.f_samp
        
        # --- The "Basis Noise" Generation ---
        # Group noise sources by their spectral shape (alpha)
        noise_params = {
            'laser_frequency': {'asd': sim_config.f_n, 'alpha': 1.0},
            'amplitude': {'asd': sim_config.amp_n, 'alpha': 0.01},
            'df': {'asd': sim_config.df_n, 'alpha': 0.01},
            'armlength': {'asd': sim_config.arml_mod_n, 'alpha': 1.0}
        }
        
        # This dictionary will hold the single, unit-variance basis noise
        # for each unique alpha value.
        basis_noises = {}
        
        # Use a different seed for each spectral shape to ensure independence
        seed_counter = 1 + trial_num * len(noise_params) # Ensures unique seeds across trials
        
        for name, params in noise_params.items():
            alpha_val = params['alpha']
            # If we haven't already generated noise for this alpha, do it now.
            if alpha_val not in basis_noises and params['asd'] != 0:
                logging.info(f"Generating basis noise for alpha = {alpha_val}...")
                generator = pyplnoise.AlphaNoise(fs, fs / num_samples, fs / 2, alpha=alpha_val, seed=seed_counter)
                # Generate one unit-variance time series
                basis_noises[alpha_val] = generator.get_series(num_samples)
                seed_counter += 1

        # --- Scale the Basis Noises to Create Final Noise Arrays ---
        final_noise = {}
        for name, params in noise_params.items():
            asd = params['asd']
            alpha_val = params['alpha']
            if asd != 0 and alpha_val in basis_noises:
                # Scale the unit-variance basis noise by the required ASD.
                # The 1/sqrt(2) factor is part of the ASD definition.
                final_noise[name] = asd / np.sqrt(2) * basis_noises[alpha_val]
            else:
                final_noise[name] = 0.0 # Assign zero if ASD is zero
                
        return final_noise

    def _vectorized_downsample(self, signal, R):
        """
        Vectorized downsampling function using NumPy's reshape and mean.
        """
        # Trim the signal so its length is a multiple of R
        trimmed_len = (len(signal) // R) * R
        trimmed_signal = signal[:trimmed_len]
        # Reshape into a 2D array and take the mean along the new axis
        return trimmed_signal.reshape(-1, R).mean(axis=1)

    def _run_static_simulation(self, sim_config, time_axis, noise_arrays):
        """
        Core physics engine for static simulation. Returns the signal array.
        Uses the simplified/approximated DFMI signal model.
        """
        # Unpack parameters for clarity
        A = sim_config.amp
        C = sim_config.visibility
        tau_r = sim_config.ref_arml / sc.c
        tau_m = sim_config.meas_arml / sc.c
        
        # Calculate modulation depth with noise
        m_noisy = 2 * np.pi * (sim_config.df + noise_arrays['df']) * (tau_m - tau_r)
        
        # Calculate total phase
        phitot = sim_config.phi + m_noisy * np.cos(2 * np.pi * sim_config.f_mod * time_axis + sim_config.psi)
        # Add laser frequency noise contribution
        phitot += 2 * np.pi * noise_arrays['laser_frequency'] * (tau_m - tau_r)
        
        # Calculate final signal
        y = A + noise_arrays['amplitude'] + A * C * np.cos(phitot)
        
        return y

    def _run_dynamic_simulation(self, sim_config, time_axis, noise_arrays):
        """
        Core physics engine for dynamic simulation. Returns signal arrays.
        Uses the full, non-approximated DFMI signal model.
        """
        # --- 1. Pre-calculate common terms ---
        omega_0_noisy = 2 * np.pi * (sc.c / sim_config.wavelength + noise_arrays['laser_frequency'])
        omega_mod = 2 * np.pi * sim_config.f_mod
        df_noisy = sim_config.df + noise_arrays['df']
        
        # --- 2. Calculate main channel parameters and phase ---
        tau_r = sim_config.ref_arml / sc.c
        tau_m = sim_config.meas_arml / sc.c
        
        # Time-varying path length difference
        tau_dl = (0.5 * sim_config.arml_mod_amp * np.sin(2 * np.pi * sim_config.arml_mod_f * time_axis + sim_config.arml_mod_psi)
                  + noise_arrays['armlength'] + sim_config.phi * sim_config.wavelength / (2 * np.pi)) / sc.c
        
        # These are the fundamental building blocks of the phase modulation
        sin_term_meas = np.sin(omega_mod * (time_axis - tau_m - tau_dl) + sim_config.psi)
        sin_term_ref = np.sin(omega_mod * (time_axis - tau_r) + sim_config.psi)
        
        # Calculate total phase for the main channel
        phi_static = omega_0_noisy * (tau_m - tau_r + tau_dl)
        phi_mod = (df_noisy / sim_config.f_mod) * (sin_term_meas - sin_term_ref)
        phitot_main = phi_static + phi_mod
        
        # Final signal for the main channel
        y_main = sim_config.amp + noise_arrays['amplitude'] + sim_config.amp * sim_config.visibility * np.cos(phitot_main)
        
        # --- 3. Calculate reference channel signal (if requested) ---
        y_ref = None
        if sim_config.refifo_opd_factor != 0:
            # REUSE common terms, only recalculate what's different
            tau_r_ref = tau_r * sim_config.refifo_opd_factor
            tau_m_ref = tau_m * sim_config.refifo_opd_factor
            
            # Recalculate only the delayed sine terms for the reference IFO
            sin_term_meas_ref = np.sin(omega_mod * (time_axis - tau_m_ref) + sim_config.psi)
            sin_term_ref_ref = np.sin(omega_mod * (time_axis - tau_r_ref) + sim_config.psi)

            phi_static_ref = omega_0_noisy * (tau_m_ref - tau_r_ref)
            phi_mod_ref = (df_noisy / sim_config.f_mod) * (sin_term_meas_ref - sin_term_ref_ref)
            phitot_ref = phi_static_ref + phi_mod_ref
            
            y_ref = sim_config.amp + noise_arrays['amplitude'] + sim_config.amp * sim_config.visibility * np.cos(phitot_ref)

        # Also return the ground-truth phase for analysis
        phi_s_ground_truth = 2 * np.pi * (sc.c / sim_config.wavelength) * (tau_m - tau_r + tau_dl)

        return y_main, y_ref, phitot_main, phi_s_ground_truth

    def simulate(self, label, n_buffers=None, n_seconds=None, simulate="dynamic", ref_channel=False, trial_num=0):
        """
        Main entry point for running a simulation.
        
        This method prepares the time axis and parameters, calls the appropriate
        simulation engine (_run_static_simulation or _run_dynamic_simulation),
        and populates the resulting DeepRawObject(s).
        """
        if label not in self.sims:
            logging.error(f"Invalid simulation label: '{label}' !!"); return

        sim_config = self.sims[label]
        
        # --- 1. Prepare Time Axis ---
        if n_buffers is not None:
            num_samples = int(n_buffers * sim_config.fit_n * sim_config.f_samp / sim_config.f_mod)
        elif n_seconds is not None:
            num_samples = int(n_seconds * sim_config.f_samp)
        else:
            num_samples = sim_config.N
            
        time_axis = np.arange(num_samples) / sim_config.f_samp
        sim_config.N = len(time_axis)

        # --- 2. Run Simulation ---
        t0 = time.time()
        logging.info(f"Simulating '{label}' ({simulate}) for {len(time_axis)/sim_config.f_samp:.2f} seconds...")
        
        noise = self._generate_noise_arrays(sim_config, time_axis, trial_num=trial_num)
        
        if simulate == "dynamic":
            y_main, y_ref, phitot, phi_s = self._run_dynamic_simulation(sim_config, time_axis, noise)
        else: # static
            y_main = self._run_static_simulation(sim_config, time_axis, noise)
            y_ref, phitot, phi_s = None, None, None # Not available in static sim

        t1 = time.time()
        sim_config.simtime = t1 - t0
        logging.info(f"Simulation finished in {sim_config.simtime:.3f} s.")

        # --- 3. Create and store result objects ---
        # Main channel
        raw_main = DeepRawObject(data=pd.DataFrame(y_main, columns=["ch0"]))
        raw_main.label, raw_main.f_samp, raw_main.f_mod = label, sim_config.f_samp, sim_config.f_mod
        raw_main.sim = sim_config
        # Store ground truth signals for analysis
        raw_main.phi = phitot
        raw_main.phi_sim = phi_s
        if phi_s is not None:
            raw_main.phi_sim_downsamp = self._vectorized_downsample(phi_s, sim_config.R)
        raw_main.f_noise, raw_main.a_noise, raw_main.l_noise, raw_main.df_noise = (
            noise['laser_frequency'], noise['amplitude'], noise['armlength'], noise['df'])
        self.raws[label] = raw_main

        # Reference channel (if created)
        if ref_channel and y_ref is not None:
            ref_label = label + '_ref'
            raw_ref = DeepRawObject(data=pd.DataFrame(y_ref, columns=["ch0"]))
            raw_ref.label, raw_ref.f_samp, raw_ref.f_mod = ref_label, sim_config.f_samp, sim_config.f_mod
            
            # Create a new, distinct sim config for the reference channel
            ref_sim_config = copy.deepcopy(sim_config)
            ref_sim_config.label = ref_label
            ref_sim_config.ref_arml *= sim_config.refifo_opd_factor
            ref_sim_config.meas_arml *= sim_config.refifo_opd_factor
            # Recalculate 'm' for the new arm lengths
            ref_sim_config.m = 2 * np.pi * ref_sim_config.df * (ref_sim_config.meas_arml - ref_sim_config.ref_arml) / sc.c
            self.sims[ref_label] = ref_sim_config
            raw_ref.sim = ref_sim_config
            
            # Store noise signals (they are common to both channels)
            raw_ref.f_noise, raw_ref.a_noise = noise['laser_frequency'], noise['amplitude']
            self.raws[ref_label] = raw_ref

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
            
        if verbose:
            # (Your logging info block here)
            pass

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
            
        if n_cores is None:
            n_cores = os.cpu_count()
        n_cores = min(n_cores, nbuf) # Don't use more cores than buffers

        if verbose:
            logging.info(f"Processing in parallel with {n_cores} cores...")
            # (Your logging info block here)
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
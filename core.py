from .plotting import *
from .fit import *
from .data import *

import ctypes as ct
import numpy as np
import scipy.constants as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pyplnoise
import time
import copy

import logging
logging.basicConfig(
format='%(asctime)s %(levelname)-8s %(message)s',
level=logging.INFO,
datefmt='%Y-%m-%d %H:%M:%S'
)


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


    def _simulate_dynamic(self, label, time, ref_channel=False):
        """
        Simulate an unequal-armlength interferometer with
        frequency-modulated input and armlength modulation

        Noise sources:
            - laser frequency noise (1/f noise)
            - additive amplitude noise (white noise)
            - laser frequency modulation amplitude noise (white noise)
            - armlength modulation amplitude noise (white noise)
        """

        def alpha(fs, size, asd, alpha=1, seed=1):
            generator = pyplnoise.AlphaNoise(fs, fs/size, fs/2, alpha=alpha, seed=seed)
            return asd / np.sqrt(2) * generator.get_series(size)

        def downsample(signal, R):
            nbuf = int(len(signal)/R)
            signal_downsampled = np.zeros(nbuf)
            for b in range(nbuf):
                signal_downsampled[b] = np.mean(signal[range(b*R,(b+1)*R)])
            return signal_downsampled

        with tqdm(total=6) as pbar:

            if self.sims[label].f_n != 0:
                laser_frequency_noise = alpha(self.sims[label].f_samp, len(time), self.sims[label].f_n, alpha=1)
            else:
                laser_frequency_noise = 0.0

            pbar.update(1)

            if self.sims[label].amp_n != 0:
                amplitude_noise = alpha(self.sims[label].f_samp, len(time), self.sims[label].amp_n, alpha=0.01)
            else:
                amplitude_noise = 0.0

            pbar.update(1)

            if self.sims[label].df_n != 0:
                df_noise = alpha(self.sims[label].f_samp, len(time), self.sims[label].df_n, alpha=0.01)
            else:
                df_noise = 0.0

            pbar.update(1)

            if self.sims[label].arml_mod_n != 0:
                arml_noise = alpha(self.sims[label].f_samp, len(time), self.sims[label].arml_mod_n, alpha=1)
            else:
                arml_noise = 0.0

            pbar.update(1)

            omega_0 = 2*np.pi*(sc.c/self.sims[label].wavelength + laser_frequency_noise)
            omega_tm = 2*np.pi*self.sims[label].arml_mod_f

            tau_r = self.sims[label].ref_arml/sc.c
            tau_m = self.sims[label].meas_arml/sc.c

            tau_dl = (0.5*self.sims[label].arml_mod_amp*np.sin(omega_tm*time + self.sims[label].arml_mod_psi)\
                + arml_noise + self.sims[label].phi * (self.sims[label].wavelength) / (2*np.pi)) / sc.c
            
            phi = omega_0*(tau_m - tau_r + tau_dl)
            phi_s = 2*np.pi*(sc.c/self.sims[label].wavelength)*(tau_m - tau_r + tau_dl)

            phitot = phi + ((self.sims[label].df+df_noise)/self.sims[label].f_mod) \
            * (np.sin(2*np.pi*self.sims[label].f_mod*(time-tau_m-tau_dl) + self.sims[label].psi) - 
                np.sin(2*np.pi*self.sims[label].f_mod*(time-tau_r) + self.sims[label].psi) )

            y = self.sims[label].amp + amplitude_noise + self.sims[label].amp*self.sims[label].visibility*np.cos(phitot)

            if ref_channel:
                tau_r = self.sims[label].ref_arml*self.sims[label].refifo_opd_factor/sc.c
                tau_m = self.sims[label].meas_arml*self.sims[label].refifo_opd_factor/sc.c
                phiref = omega_0*(tau_m - tau_r) + ((self.sims[label].df+df_noise)/self.sims[label].f_mod) \
                * (np.sin(2*np.pi*self.sims[label].f_mod*(time-tau_m) + self.sims[label].psi) - 
                    np.sin(2*np.pi*self.sims[label].f_mod*(time-tau_r) + self.sims[label].psi) )
                y_ref = self.sims[label].amp + amplitude_noise + self.sims[label].amp*self.sims[label].visibility*np.cos(phiref)

            pbar.update(1)
            raw = DeepRawObject(data=pd.DataFrame(y, columns=["ch0"]))
            raw.label = label
            raw.f_samp = self.sims[label].f_samp
            raw.f_mod = self.sims[label].f_mod
            raw.phi = phitot
            raw.phi_sim = phi_s
            raw.phi_sim_downsamp = downsample(phi_s, self.sims[label].R)
            raw.f_noise = laser_frequency_noise
            raw.a_noise = amplitude_noise
            raw.l_noise = arml_noise
            raw.df_noise = df_noise
            self.raws[raw.label] = raw

            if ref_channel:
                raw = DeepRawObject(data=pd.DataFrame(y_ref, columns=["ch0"]))
                raw.label = label + '_ref'
                raw.f_samp = self.sims[label].f_samp
                raw.f_mod = self.sims[label].f_mod
                raw.phi = phitot
                raw.phi_sim = phi_s
                raw.phi_sim_downsamp = downsample(phi_s, self.sims[label].R)
                raw.f_noise = laser_frequency_noise
                raw.a_noise = amplitude_noise
                raw.l_noise = arml_noise
                raw.df_noise = df_noise
                self.raws[raw.label] = raw
                self.sims[raw.label] = copy.deepcopy(self.sims[label])
                self.sims[raw.label].label = raw.label
                self.sims[raw.label].ref_arml = self.sims[label].ref_arml*self.sims[label].refifo_opd_factor
                self.sims[raw.label].meas_arml = self.sims[label].meas_arml*self.sims[label].refifo_opd_factor
                self.sims[raw.label].m = self.sims[label].m*self.sims[label].refifo_opd_factor

            pbar.update(1)
            pbar.close()


    def _simulate_static(self, label, time):
        """
        Simulate an unequal-armlength interferometer with 
        frequency-modulated input and static arms

        Noise sources:
            - laser frequency noise (1/f noise)
            - additive amplitude noise (white noise)
            - laser frequency modulation amplitude noise (white noise)
        """

        def alpha(fs, size, asd, alpha=1, seed=1):
            generator = pyplnoise.AlphaNoise(fs, fs/size, fs/2, alpha=alpha, seed=seed)
            return asd / np.sqrt(2) * generator.get_series(size)

        with tqdm(total=5) as pbar:

            if self.sims[label].f_n != 0:
                laser_frequency_noise = alpha(self.sims[label].f_samp, len(time), self.sims[label].f_n, alpha=1)
            else:
                laser_frequency_noise = 0.0

            pbar.update(1)

            if self.sims[label].amp_n != 0:
                amplitude_noise = alpha(self.sims[label].f_samp, len(time), self.sims[label].amp_n, alpha=0.01)
            else:
                amplitude_noise = 0.0

            pbar.update(1)

            if self.sims[label].df_n != 0:
                df_noise = alpha(self.sims[label].f_samp, len(time), self.sims[label].df_n, alpha=0.01)
            else:
                df_noise = 0.0

            pbar.update(1)

            A = self.sims[label].amp
            C = self.sims[label].visibility
            tau_r = self.sims[label].ref_arml/sc.c
            tau_m = self.sims[label].meas_arml/sc.c
            m = 2*np.pi*(self.sims[label].df+df_noise)*(tau_m - tau_r)

            phitot = self.sims[label].phi + m * np.cos(2*np.pi*self.sims[label].f_mod*time + self.sims[label].psi)
            phitot = phitot + 2*np.pi*laser_frequency_noise*(tau_m - tau_r)

            y = A + amplitude_noise + A*C*np.cos(phitot)
            pbar.update(1)

            raw = DeepRawObject(data=pd.DataFrame(y, columns=["ch0"]))
            raw.label = label
            raw.f_samp = self.sims[label].f_samp
            raw.f_mod = self.sims[label].f_mod
            raw.f_noise = laser_frequency_noise
            raw.a_noise = amplitude_noise
            raw.df_noise = df_noise
            self.raws[raw.label] = raw

            pbar.update(1)
            pbar.close()


    def simulate(self, label, n_buffers=None, n_seconds=None, simulate="dynamic", ref_channel=False):
        if label not in self.sims:
            logging.error('Invalid label !!')
            return

        self.sims[label].fit_n = self.sims[label]._fit_n

        if n_buffers is not None:
            taxis = np.arange(n_buffers*self.sims[label].fit_n*self.sims[label].f_samp/self.sims[label].f_mod) / self.sims[label].f_samp
        elif n_seconds is not None:
            taxis = np.arange(n_seconds*self.sims[label].f_samp) / self.sims[label].f_samp
        else:
            taxis = np.arange(self.sims[label].N) / self.sims[label].f_samp

        self.sims[label].N = len(taxis)

        if simulate == "dynamic":
            t0 = time.time()
            self._simulate_dynamic(label, taxis, ref_channel)
            t1 = time.time()
            self.sims[label].simtime = t1-t0
        else:
            t0 = time.time()
            self._simulate_static(label, taxis)
            t1 = time.time()
            self.sims[label].simtime = t1-t0


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


    def _fit_(self, label, b, R, ndata, fit_label):
        """Routine called by DeepFMFramework.fit().
        Invokes the fit() function declared in dfmfit.py.
        Acts on the raw data of channel `c` in the range
        determined by the buffer number `b`.

        Returns a DataFrame containing the fit parameters,
        including the corresponding channel and buffer number.
        """
        QI_data_mean = np.zeros(2*ndata)

        point = self.fits_df[fit_label][self.fits_df[fit_label]['b']==b-1]

        fitparm = np.zeros(4)

        fitparm[0] = point['amp'].iloc[0]
        fitparm[1] = point['m'].iloc[0]
        fitparm[2] = point['phi'].iloc[0]
        fitparm[3] = point['psi'].iloc[0]
        retssq = point['ssq'].iloc[0]

        buf = range(b*R,(b+1)*R)

        raw_buffer = np.array(self.raws[label].data.loc[buf]).flatten()

        w0 = 2. * np.pi * self.raws[label].f_mod / self.raws[label].f_samp

        for n in range(ndata):
            Q_data, I_data = calculate_quadratures(n, raw_buffer, w0, R)
            QI_data_mean[n] = Q_data.mean()
            QI_data_mean[n+ndata] = I_data.mean()
            vec_sig = QI_data_mean[n] + 1j*QI_data_mean[n+ndata]
            phase = np.angle(vec_sig)
            amp = abs(vec_sig)
            QI_data_mean[n] = amp * cos(phase)
            QI_data_mean[n+ndata] = amp * sin(phase)
            dc = float(self.raws[label].data.loc[buf].mean().iloc[0])
        # ===================================================================
        # START: Diagnostic Print Block
        # ===================================================================
        # print("\n" + "="*80)
        # print(f"--- DIAGNOSTIC: Data for fit buffer b = {b} ---")
        # print("Final I/Q data being passed to fitter:")
        # print(QI_data_mean)
        # if np.all(QI_data_mean == 0):
        #     print("\n>>> CRITICAL: All I/Q values are zero! Checking inputs to quadratures...")
        #     print(f"    Buffer size (R): {R}")
        #     print(f"    Harmonic count (ndata): {ndata}")
        #     print(f"    w0 (rad/sample): {w0}")
        #     # Let's check the raw data that went in
        #     raw_buffer = np.array(self.raws[label].data.loc[buf])
        #     print(f"    Shape of raw_buffer: {raw_buffer.shape}")
        #     print(f"    Mean of raw_buffer: {np.mean(raw_buffer)}")
        #     print(f"    Std dev of raw_buffer: {np.std(raw_buffer)}")
        # print("="*80 + "\n")
        # ===================================================================
        # END: Diagnostic Print Block
        # ===================================================================
        # print('Before: ', fitparm)
        # logging.info("Trying fit...")
        fitok, fitparm, retssq = fit(ndata, QI_data_mean, fitparm)
        # logging.info("fit returned with fitok = {}...".format(fitok))
        # print('After: ', fitparm)


        buffer_df = pd.DataFrame()
        buffer_df['amp'] = [fitparm[0]]
        buffer_df['m'] = [fitparm[1]]
        buffer_df['phi'] = [fitparm[2]]
        buffer_df['psi'] = [fitparm[3]]
        buffer_df['dc'] = [dc]
        buffer_df['ssq'] = [retssq]
        buffer_df['fitok'] = [fitok]
        buffer_df['b'] = [b]

        return buffer_df


    def fit(self, label, n=None, init_a=1.6, init_m=6.0, ndata=10, fit_label=None, init_psi=False, verbose=True):
        """Performs fit on the raw data specified by `label`, generating a corresponding
        DeepFitObject.
        """
        if n is None:
            try:
                n = self.sims[label].fit_n
            except:
                logging.error("fit_n not specified !!")
                return

        if fit_label is None:
            fit_label = label+'_fit'
        if label not in self.raws:
            logging.error('Invalid label !!')
            return
        else:
            R, fs, nbuf = self.fit_init(label, n)
            nbuf = int(self.raws[label].data.shape[0]/R)
            if nbuf == 0:
                logging.error('Check buffer size !!')
            if verbose:
                logging.info(\
                    """
                    New DeepFitObject
                    Label: {}
                    Start time: {}
                    Sampling frequency: {}
                    Modulation frequency: {}
                    Downsampling factor: {}
                    n: {}
                    Fit data rate: {}
                    Buffer count: {}""".replace('\t', '').format(fit_label, self.raws[label].t0, self.raws[label].f_samp, self.raws[label].f_mod, R, n, fs, nbuf))


        if init_psi:
            psi = self.psi_init(label, init_psi, init_a, init_m, R, ndata, fit_label, verbose)
        else:
            psi = 0.0

        self.fits_df[fit_label] = pd.DataFrame(\
        [{'amp': init_a, 'm': init_m, 'phi': 0, 'psi': psi, 'dc': 1, 'ssq': 0, 'fitok': 0, 'b':-1}])

        if verbose:
            logging.info('Processing...')
            for b in tqdm(range(nbuf)):
                newfit = self._fit_(label, b, R, ndata, fit_label);
                self.fits_df[fit_label] = pd.concat((self.fits_df[fit_label], newfit), ignore_index=True)
            logging.info('Done!!!')
        else:
            for b in range(nbuf):
                newfit = self._fit_(label, b, R, ndata, fit_label);
                self.fits_df[fit_label] = pd.concat((self.fits_df[fit_label], newfit), ignore_index=True)

        self.fits_df[fit_label] = self.fits_df[fit_label][self.fits_df[fit_label]['b'] > -1]
        self.fits_df[fit_label] = self.fits_df[fit_label].reset_index()
        self.fits_df[fit_label] = self.fits_df[fit_label].drop('index', axis=1)

        fit = DeepFitObject()
        fit.n = n
        fit.R = R
        fit.fs = fs
        fit.nbuf = nbuf
        fit.ndata = ndata
        fit.init_a = init_a
        fit.init_m = init_m
        fit.t0 = self.raws[label].t0
        fit.f_samp = self.raws[label].f_samp
        fit.f_mod  = self.raws[label].f_mod
        fit.ssq = np.array(self.fits_df[fit_label]['ssq'])
        fit.amp = np.array(self.fits_df[fit_label]['amp'])
        fit.m   = np.array(self.fits_df[fit_label]['m'])
        fit.phi = np.array(self.fits_df[fit_label]['phi'])
        fit.psi = np.array(self.fits_df[fit_label]['psi'])
        fit.dc  = np.array(self.fits_df[fit_label]['dc'])
        fit.time = np.arange(0, fit.ssq.shape[0]/fit.fs, 1./fit.fs)
        fit.label = fit_label
        self.fits[fit_label] = fit
        return fit


    def psi_init(self, label, method, init_a, init_m, R, ndata, fit_label, verbose):

        if verbose:
            logging.info("Initializing psi parameter using {} method".format(method))

        if method == 'scan':
            init_psi = 0.0
            best_ssq = 9e99
            for psi in np.linspace(0, 2*np.pi, 20):
                new_ssq = self.try_psi(psi, label, init_a, init_m, R, ndata, fit_label)
                if new_ssq < best_ssq:
                    best_ssq = new_ssq
                    init_psi = psi

        if method == 'minimize':
            from scipy.optimize import minimize
            init_psi = 0.0
            best_ssq = 9e99
            for psi in np.linspace(0, 2*np.pi, 20):
                new_ssq = self.try_psi(psi, label, init_a, init_m, R, ndata, fit_label)
                if new_ssq < best_ssq:
                    best_ssq = new_ssq
                    init_psi = psi
            logging.info("Minimizer start with initial guess psi = {}".format(init_psi))
            res = minimize(self.try_psi, init_psi, args=(label, init_a, init_m, R, ndata, fit_label),\
            method="Nelder-Mead", bounds=[(0.0,2*np.pi)], options={"xatol": 1e-9, "fatol": 1e-14, "disp": True})
            init_psi = res.x[0]

        if verbose:
            logging.info("Selected init_psi = {}".format(init_psi))

        return init_psi


    def try_psi(self, psi, label, init_a, init_m, R, ndata, fit_label):

        self.fits_df[label] = pd.DataFrame(\
            [{'amp': init_a, 'm': init_m, 'phi': 0, 'psi': psi, 'dc': 1, 'ssq': 0, 'fitok': 0, 'b':-1}])
        newfit = self._fit_(label, 0, R, ndata, fit_label);
        return newfit['ssq'][0]


    def pfit(self, n, labels=None, init_a=1.6, init_m=6.0, ndata=10):
        """Performs fit on the raw data specified by `labels`, generating the corresponding
        DeepFitObject's, parallelizing the computation over the number of raw data channels
        selected.

        Broken as of Python 3.9
        
        https://stackoverflow.com/questions/62830911/typeerror-cannot-pickle-weakref-object
        """
        from multiprocessing import Pool
        from functools import partial

        if labels == None:
            labels = list(self.raws.keys())

        if len(labels) == 1:
            logging.info('For fitting a single raw data channel, use fit()...')
            return

        nraws = len(labels)
        cpu_count = os.cpu_count()
        threads = nraws

        if cpu_count <= nraws:
            threads = cpu_count

        logging.info('Starting parallel computation with {} threads...'.format(threads))

        with Pool(threads) as p:
            fits = list(tqdm(p.imap(partial(\
                self.fit, n=n, init_a=init_a, init_m=init_m, ndata=ndata, fit_label=None, verbose=False), labels), total=len(labels)))
        
        for fit in fits:
            self.fits[fit.label] = fit

        logging.info('Done!!')


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
                ax6.semilogy(self.fits[k].time, self.fits[k].ssq);
                ax5.plot(self.fits[k].time, self.fits[k].dc);
                ax4.plot(self.fits[k].time, self.fits[k].amp);
                ax3.plot(self.fits[k].time, self.fits[k].m);
                ax2.plot(self.fits[k].time, self.fits[k].phi);
                ax1.plot(self.fits[k].time, self.fits[k].psi, label=str(k));
        else:
            for k in labels:
                ax6.semilogy(self.fits[k].time, self.fits[k].ssq);
                ax5.plot(self.fits[k].time, self.fits[k].dc);
                ax4.plot(self.fits[k].time, self.fits[k].amp);
                ax3.plot(self.fits[k].time, self.fits[k].m);
                ax2.plot(self.fits[k].time, self.fits[k].phi);
                ax1.plot(self.fits[k].time, self.fits[k].psi, label=str(k));

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

        ax6.semilogy(self.fits[label1].time[:c], self.fits[label1].ssq[:c] - self.fits[label2].ssq[:c]);
        ax5.plot(self.fits[label1].time[:c], self.fits[label1].dc[:c]      - self.fits[label2].dc[:c]);
        ax4.plot(self.fits[label1].time[:c], self.fits[label1].amp[:c]     - self.fits[label2].amp[:c]);
        ax3.plot(self.fits[label1].time[:c], self.fits[label1].m[:c]       - self.fits[label2].m[:c]);
        ax2.plot(self.fits[label1].time[:c], self.fits[label1].phi[:c]     - self.fits[label2].phi[:c]);
        ax1.plot(self.fits[label1].time[:c], self.fits[label1].psi[:c]     - self.fits[label2].psi[:c], label=label1+'-'+label2);

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

        ax6.semilogy(self.fits[label1].time, self.fits[label1].ssq);
        ax5.plot(self.fits[label1].time, self.fits[label1].dc);
        ax4.plot(self.fits[label1].time, self.fits[label1].amp);
        ax3.plot(self.fits[label1].time, self.fits[label1].m);
        ax2.plot(self.fits[label1].time, self.fits[label1].phi);
        ax1.plot(self.fits[label1].time, self.fits[label1].psi, label=label1);

        ax6.semilogy(self.fits[label2].time, self.fits[label2].ssq, linestyle='dashed');
        ax5.plot(self.fits[label2].time, self.fits[label2].dc, linestyle='dashed');
        ax4.plot(self.fits[label2].time, self.fits[label2].amp, linestyle='dashed');
        ax3.plot(self.fits[label2].time, self.fits[label2].m, linestyle='dashed');
        ax2.plot(self.fits[label2].time, self.fits[label2].phi, linestyle='dashed');
        ax1.plot(self.fits[label2].time, self.fits[label2].psi, linestyle='dashed', label=label2);

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
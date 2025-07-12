from .fit import *
from .plotting import *

import numpy as np
import scipy.constants as sc
import pandas as pd
import matplotlib.pyplot as plt

import logging
logging.basicConfig(
format='%(asctime)s %(levelname)-8s %(message)s',
level=logging.INFO,
datefmt='%Y-%m-%d %H:%M:%S'
)

class DeepRawObject():
    """Structure of data containing the raw data for a single channel,
    as well as some important metadata.
    """

    def __init__(self, data=None):
        self.raw_file = None       # The raw data file
        self.label = None          # Label of the data
        self.t0 = None             # Start time
        self.f_samp = None         # Sampling frequency
        self.f_mod = None          # Modulation frequency
        self.sim = None            # Associated DFMIObject
        self.data = pd.DataFrame() # The raw data

        """
        Simulation outputs
        """
        self.phi = None              # IFO phase from dynamic simulations, incl. laser frequency noise
        self.phi_sim = None          # IFO phase from dynamic simulations, not incl. laser frequency noise
        self.phi_sim_downsamp = None # phi_sim downsampled
        self.f_noise = None          # Laser frequency noise
        self.l_noise = None          # Armlength modulation noise
        self.a_noise = None          # Signal amplitude noise
        self.df_noise = None         # Laser frequency modulation amplitude noise

        if data is not None:
            self.data = data


    def info(self):
        """Prints useful info about the loaded fit data.
        """

        logging.info(\
"""
DeepRawObject
Label: {}
Start time: {}
Sampling frequency: {}
Modulation frequency: {}
"""\
            .format(self.label, self.t0, self.f_samp, self.f_mod))


    def to_txt(self, filename):
        """Save the fit to a txt file following the DFMSWPM fit_data format 
        """

        lines = ['% fit_data'.format(),\
        '% Message goes here',\
        '% Number of channels: {}'.format(1),\
        '% Start time: {}'.format(self.t0),\
        '% Sampling frequency: {}'.format(self.f_samp),\
        '% Modulation frequency: {}'.format(self.f_mod),\
        '% n: {}'.format(0),\
        '% Downsampling factor: {}'.format(0),\
        '% Fit data rate: {}'.format(0),\
        '% Initial amplitude: {}'.format(0),\
        '% Initial modulation depth: {}'.format(0),\
        '%',\
        'ch0 ']

        with open(filename, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
            for l in range(len(self.data)):
                line = str(self.data.iloc[0][0]) + ' '
                f.write(line)
                f.write('\n')


    def parse_header(self):
        """Parse the header of a raw_data file.

        This function must be revised should the format of the header change.
        """

        lines = []
        values = []
        res = {}

        if self.raw_file is not None:
            with open(self.raw_file) as f:
                for _ in range(11):
                    lines.append(f.readline())
        else:
            logging.error('No file specified !!')

        for v in range(2,11):
            values.append(''.join([c for c in lines[v] if c in '1234567890.']))

        self.t0     =   int(values[0])
        self.f_samp = float(values[1])
        self.f_mod  = float(values[2])

    def plot(self, title=None, ax=None, xrange=None, figsize=(20,5), *args, **kwargs):
        
        t_list = [np.arange(len(self.data))/self.f_samp]
        y_list = [self.data]
        
        return time_plot(t_list, y_list, label_list=[self.label], ax=ax, xrange=xrange,\
            title=title, y_label='Voltage(t)', figsize=figsize, remove_y_offsets=False, remove_time_offsets=False, *args, **kwargs)


class DeepFitObject():
    """Structure of data containing the fit parameters for a single channel,
    as well as some important metadata.
    """

    def __init__(self):
        self.fit_file = None       # The fit data file
        self.label = None          # Label of the data
        self.n = None              # n*f_samp/f_mod = bufferSize; data_rate = f_samp/bufferSize = f_mod/n;
        self.t0 = None             # Start time  
        self.R = None              # Downsampling factor
        self.fs = None             # Fit data rate
        self.f_samp = None         # Sampling frequency
        self.f_mod = None          # Modulation frequency
        self.ndata = 10            # Number of higher harmonics to fit
        self.init_a = 1.6          # Initial value of the amplitude
        self.init_m = 6.0          # Initial value of the effective modulation index
        self.nbuf = None           # Buffers contained in raw data
        self.time = np.array([])   # Fit time
        self.ssq  = np.array([])   # Fit sum of squares
        self.amp  = np.array([])   # Fit amplitude
        self.m    = np.array([])   # Fit effective modulation index
        self.tau  = np.array([])   # Fit time delay (L/c)
        self.phi  = np.array([])   # Fit interferometric phase
        self.psi  = np.array([])   # Fit modulation phase
        self.dc   = np.array([])   # Fit dc level
        self.f = None              # LPSD frequency
        self.Sxx = None            # LPSD
        self.olap = "default"      # LPSD calculation parameters
        self.bmin = 1
        self.Lmin = 0
        self.Jdes = 500
        self.Kdes = 100
        self.order = 0
        self.win = np.kaiser
        self.psll = 200
        self.fig = None


    def info(self):
        """Prints useful info about the loaded fit data.
        """

        logging.info(\
"""
DeepFitObject
Label: {}
Start time: {}
Sampling frequency: {}
Modulation frequency: {}
Downsampling factor: {}
n: {}
Fit data rate: {}
"""\
            .replace('\t', '').format(self.label, self.t0, self.f_samp, self.f_mod, self.R, self.n, self.fs))


    def to_txt(self, filename):
        """Save the fit to a txt file following the DFMSWPM fit_data format 
        """

        lines = ['% fit_data'.format(),\
        '% Message goes here',\
        '% Number of channels: {}'.format(1),\
        '% Start time: {}'.format(self.t0),\
        '% Sampling frequency: {}'.format(self.f_samp),\
        '% Modulation frequency: {}'.format(self.f_mod),\
        '% n: {}'.format(int(self.n)),\
        '% Downsampling factor: {}'.format(int(self.R)),\
        '% Fit data rate: {}'.format(self.fs),\
        '% Initial amplitude: {}'.format(self.init_a),\
        '% Initial modulation depth: {}'.format(self.init_m),\
        '%',\
        'ssq0 amp0 m0 phi0 psi0 dc0 ']

        with open(filename, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
            for l in range(len(self.ssq)):
                line = str(self.ssq[l]) + ' ' \
                + str(self.amp[l]) + ' ' \
                + str(self.m[l]) + ' ' \
                + str(self.phi[l]) + ' ' \
                + str(self.psi[l]) + ' ' \
                + str(self.dc[l]) + ' '
                f.write(line)
                f.write('\n')


    def parse_header(self):
        """Parse the header of a fit_data file.

        This function must be revised should the format of the header change.
        """

        lines = []
        values = []
        res = {}

        if self.fit_file is not None:
            with open(self.fit_file) as f:
                for _ in range(11):
                    lines.append(f.readline())
        else:
            logging.error('No file specified !!')

        for v in range(2,11):
            values.append(''.join([c for c in lines[v] if c in '1234567890.']))

        self.t0     =   int(values[0])
        self.f_samp = float(values[1])
        self.f_mod  = float(values[2])
        self.n      =   int(values[3])
        self.R      =   int(values[4])
        self.fs     = float(values[5])


    def calc_lpsd(self):
        """Calculate LPSD of interferometric phase parameter (phi).
        """
        from spectools.lpsd import lpsd
        self.f, _, self.Sxx, _, _, _ = lpsd(self.phi, \
            self.fs, self.olap, self.bmin, self.Lmin, self.Jdes, self.Kdes, self.order, self.win, self.psll, return_type='legacy')


    def plot_lpsd(self, nm=True, pm=True):
        """Plot the LPSD of interferometric phase parameter (phi).
        """
        if self.f == None:
            logging.info('Calculating the LPSD now...')
            self.calc_lpsd()

        title = self.label
        xlabel = r"Frequency$\,({\mathrm{Hz}})$"
        ylabel = r"Phase ASD$\,(\mathrm{rad}/\sqrt{\mathrm{Hz}})$"

        fig, ax = plt.subplots(1, figsize=figsize(1.2), dpi=1 * 300)

        log_plot(self.f, np.sqrt(self.Sxx), ax, title, xlabel, ylabel, self.label)

        if nm == True:
            log_plot(self.f, displacement_req(self.f, 1e-9, 3e-3), ax, title, xlabel, ylabel, "1 nm")

        if pm == True:
            log_plot(self.f, displacement_req(self.f, 1e-12, 3e-3), ax, title, xlabel, ylabel, "1 pm")


    def plot(self, xrange=None, timeaxis=True):
        """Plot the existing fit data in the specified `xrange`,
        or in the entire range.
        """
        self.fig, (ax1, ax2, ax3, ax4, ax5, ax6) = dfm_axes()

        if timeaxis:
            ax6.set_xlabel('Time (s)')
            ax6.semilogy(self.time, self.ssq)
            ax5.plot(self.time, self.dc)
            ax4.plot(self.time, self.amp)
            ax3.plot(self.time, self.m)
            ax2.plot(self.time, self.phi)
            ax1.plot(self.time, self.psi, label=self.label)
        else:
            ax6.set_xlabel('Buffer count')
            ax6.semilogy(self.ssq)
            ax5.plot(self.dc)
            ax4.plot(self.amp)
            ax3.plot(self.m)
            ax2.plot(self.phi)
            ax1.plot(self.psi, label=self.label)

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



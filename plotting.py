import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates


default_rc = {
    'figure.dpi': 150,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid': True,
    'grid.color': '#FFD700',
    'grid.linewidth': 0.7,
    'grid.linestyle': '--',
    'axes.prop_cycle': plt.cycler('color', [
        '#000000', '#DC143C', '#00BFFF', '#FFD700', '#32CD32',
        '#FF69B4', '#FF4500', '#1E90FF', '#8A2BE2', '#FFA07A', '#8B0000'
    ]),
}
# plt.rcParams.update(default_rc)

legend_params = {
    'loc': 'best',
    'fontsize': 8,
    'frameon': True,
}

def apply_legend(ax):
    legend = ax.legend(**legend_params)
    frame = legend.get_frame()
    frame.set_alpha(1.0)          # No transparency
    frame.set_edgecolor('black')  # Black border
    frame.set_linewidth(0.7)
    try:
        frame.set_boxstyle('Square')  # No rounded corners
    except AttributeError:
        pass  # Safe fallback for older matplotlib
    return legend

def figsize(scale):
    fig_width_pt = 390  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Golden ratio
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

def lin_plot(x, y, ax, title, xlabel, ylabel, label, *args, **kwargs):
    if x is not None:
        ax.plot(x, y, label=label, *args, **kwargs)
    else:
        ax.plot(y, label=label, *args, **kwargs)
    ax.set_title(title, fontsize=11)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

def log_plot(x, y, ax, title, xlabel, ylabel, label, *args, **kwargs):
    ax.loglog(x, y, label=label, *args, **kwargs)
    ax.set_title(title, fontsize=11)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

def stem_plot(x, y, ax, title, xlabel, ylabel, label, *args, **kwargs):
    ax.vlines(x, 0, y, color='b', label=label, *args, **kwargs)
    ax.set_ylim([1.05 * y.min(), 1.05 * y.max()])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

def autoscale_y(ax, margin=0.1):
    """
    This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims
    """
    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        y_displayed = yd[((xd > lo) & (xd < hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed) - margin * h
        top = np.max(y_displayed) + margin * h
        return bot, top

    lines = ax.get_lines()
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot, top)

def dfm_axes(figsize=None):
    if figsize is None:
        figsize = figsize(2)

    fig1, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,
                                                   figsize=figsize(2),
                                                   dpi=1 * 300,
                                                   sharex=True)

    ax6.xaxis.set_tick_params(labelsize=11)
    ax6.set_xlabel('Fit point')
    ax6.set_ylabel('SSQ')
    ax5.set_ylabel('DC value')
    ax4.set_ylabel('AC amp')
    ax3.set_ylabel('m')
    ax2.set_ylabel(r'$\phi\,(\mathrm{rad})$')
    ax1.set_ylabel(r'$\psi\,(\mathrm{rad})$')

    fig1.align_ylabels()

    return fig1, (ax1, ax2, ax3, ax4, ax5, ax6)

def phase_req(x, level, corner):
    y = np.zeros(len(x))
    for k, f in enumerate(x):
        y[k] = level * np.sqrt(1 + (corner / f)**4)
    return y

def displacement_req(x, level, corner):
    wl = 1064e-9
    phase_level = (2 * np.pi / wl) * level
    return phase_req(x, phase_level, corner)

def time_plot(t_list, y_list, label_list=None, ax=None, xrange=None, \
    title=None, y_label=None, figsize=(20,5),\
    remove_y_offsets=False, remove_time_offsets=False, *args, **kwargs):
    """
    Time series plot.

    :param t_list: List containing pd.DataFrame, np.array, or list of time values
    :type t_list: List
    :param y_list: List containing pd.DataFrame, np.array, or list of values
    :type y_list: List
    :param label_list: List containing strings of data labels
    :type label_list: List
    :param xrange: List of [x_min, x_max]
    :type xrange: List
    :param title: Title string
    :type title: String
    :param y_label: Label of the y-axis
    :type y_label: String
    :param figsize: Figure size in inches
    :type figsize: (float, float)
    :return: Matplotlib Figure and Axes
    :rtype: (Figure, axes.Axes)
    """
    assert len(t_list) == len(y_list), "len(y_list) != len(t_list)"
    if label_list is not None:
        assert len(t_list) == len(label_list), "len(label_list) != len(t_list)"

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.set_xlabel('Time')
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)

    if remove_y_offsets:
        for i in range(len(t_list)):
            y = y_list[i].copy() - np.average(y_list[i])
            if remove_time_offsets:
                try:
                    t = t_list[i].copy() - t_list[i].iloc[0]
                except:
                    t = np.array(t_list[i]) - t_list[i][0]
            else:
                t = t_list[i]
            if label_list is not None:
                ax.plot(t, y, label=label_list[i],  *args, **kwargs)
            else:
                ax.plot(t, y,  *args, **kwargs)
    else:
        for i in range(len(t_list)):
            if remove_time_offsets:
                try:
                    t = nt_list[i].copy() - t_list[i].iloc[0]
                except:
                    t = np.array(t_list[i]) - t_list[i][0]
            else:
                t = t_list[i]
            if label_list is not None:
                ax.plot(t, y_list[i], label=label_list[i],  *args, **kwargs)
            else:
                ax.plot(t, y_list[i],  *args, **kwargs)
    
    if xrange is not None:
        ax.set_xlim(xrange)
        autoscale_y(ax)

    fig.tight_layout()
    return ax

def time_histogram(df, time_key, y_key, time_floor='h', nbins = None,\
start=None, end=None, format_str = '%d/%m/%Y-%H:%M', figsize=(20,5)):
    """Plot a 2d histogram of time series data. The with of the time bins
    is adjusted by the ``time_floor`` parameter. Examples:
    
        'h'   : 1 hour
        '6h'  : 6 hours
        '10t' : 10 minutes
        'd'   : 1 day

    :param df: DataFrame containing time series data
    :type df: pandas.DataFrame
    :param time_key: Key of column containing the timestamp (x-axis data)
    :type time_key: str
    :param y_key: Key of column containing the time series data (y-axis data)
    :type y_key: str
    :param time_floor: Width of the time bins, defaults to 'h'
    :type time_floor: str, optional
    :param nbins: Number of bins for the y-axis (if None, the number of bins will be taken as the number of unique values of df[y_key]), defaults to None
    :type nbins: int, optional  
    :param start: A start date for the plot, defaults to None
    :type start: str, optional
    :param end: An end date for the plot, defaults to None
    :type end: str, optional
    :param format_str: Format string for time axis labels, defaults to '%d/%m/%Y-%H:%M'
    :type format_str: str, optional
    :param figsize: Figure size, defaults to (20,5)
    :type figsize: tuple, optional
    :return: Matplotlib Figure and Axes
    :rtype: (Figure, axes.Axes)
    
    """
    datetime = pd.Series()

    try:
        datetime = pd.to_datetime(df[time_key])
    except:
        print("The timestamp cannot be converted to datetime")
        exit()

    dates_of_interest = pd.Series(True, index=datetime.index)

    if (start is not None) & (end is None) :
        dates_of_interest = datetime > pd.to_datetime(start)

    if (start is None) & (end is not None) :
        dates_of_interest = datetime < pd.to_datetime(end)

    if (start is not None) & (end is not None) :
        dates_of_interest = (datetime > pd.to_datetime(start))&(datetime < pd.to_datetime(end))

    try:
        assert dates_of_interest.value_counts().loc[False] != len(dates_of_interest), "Null time axis"
    except KeyError:
        assert dates_of_interest.value_counts().loc[True] > 0, "Null time axis"

    time_axis = datetime.dt.floor(time_floor)

    assert len(time_axis) > 0, "Null time axis"

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    cmap = copy.copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))
    
    if nbins is None:
        nbins = df[dates_of_interest][y_key].unique().shape[0]

    h, xedges, yedges = np.histogram2d(\
    mdates.date2num(time_axis[dates_of_interest].values),\
    df.loc[dates_of_interest, y_key],\
    bins=[time_axis[dates_of_interest].unique().shape[0], nbins])

    pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm(), rasterized=True)
    
    fig.colorbar(pcm, ax=ax, label="Number of events", pad=0)
    
    format_ = mdates.DateFormatter(format_str)
    ax.xaxis.set_major_formatter(format_)
    
    ax.set_xlabel('Time')
    ax.set_ylabel(y_key)

    return fig, ax

def asd_plot(f_list, asd_list, label_list=None, title=None, unit=None, psd=False):
    """
    Plot spectral densities on logarithmic axes
    """
    if label_list is None:
        label_list = ['']*len(f_list)
        for i, asd in enumerate(asd_list):
            label_list[i] = str(i)

    if title is None:
        title = ''

    if unit is None:
        unit = 'A'
        
    xlabel = r"Frequency$\,({\mathrm{Hz}})$"
    ylabel = r"ASD$\,(\mathrm{" + unit + r"}/\sqrt{\mathrm{Hz}})$"
        
    fig, ax = plt.subplots(1, figsize=figsize(1.2), dpi=1 * 300)

    if psd is False:
        for i, asd in enumerate(asd_list):
            log_plot(f_list[i], asd, ax, title, xlabel, ylabel, label_list[i])
    else:
        for i, asd in enumerate(asd_list):
            log_plot(f_list[i], np.sqrt(asd), ax, title, xlabel, ylabel, label_list[i])       

    apply_legend(ax)

    return ax
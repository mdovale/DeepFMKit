from .experiments import *
from .physics import *
import DeepFMKit.core as dfm

import numpy as np
import scipy.constants as sc

def calculate_ambiguity_boundary_point(params):
    """
    Calculates the coarse phase error for a single point in the
    (delta_f, delta_m) parameter space.

    This is a simple, non-simulation worker that applies the core theoretical
    equation: delta_Phi_coarse = -delta_m * (f0 / df).

    Parameters
    ----------
    params : dict
        A dictionary containing the point's parameters:
        - 'delta_f': The modulation amplitude (df).
        - 'delta_m': The total uncertainty/bias on the m estimate.
        - 'f0': The laser carrier frequency.
        - 'grid_i', 'grid_j': Indices for the output grid.

    Returns
    -------
    tuple
        (grid_i, grid_j, coarse_phase_error)
    """
    delta_f = params['delta_f']
    delta_l = params['delta_l']
    f0 = params['f0']
    grid_i = params['grid_i']
    grid_j = params['grid_j']
    
    # Avoid division by zero if delta_f is zero
    if delta_f == 0:
        return (grid_i, grid_j, float('inf'))
        
    coarse_phase_error = -2*np.pi*(delta_l / sc.c) * (f0 / delta_f)
    
    return (grid_i, grid_j, np.abs(coarse_phase_error))

def run_single_trial(
    laser_config: LaserConfig,
    main_ifo_config: InterferometerConfig,
    fitter_method: str,
    fitter_kwargs: Optional[dict] = None,
    witness_ifo_config: Optional[InterferometerConfig] = None,
    n_seconds: Optional[float] = None,
    trial_num: int = 0
) -> Optional[dfm.DeepFitObject]:
    """Encapsulates the standard "Configure-Simulate-Fit" workflow.

    This function provides a high-level interface for running a single,
    self-contained simulation and fitting trial. It handles the boilerplate
    of instantiating the DeepFitFramework, configuring channels, simulating,
    and fitting. This allows worker functions to focus only on defining the
    physics of the trial, not the execution details.

    Parameters
    ----------
    laser_config : LaserConfig
        A configured laser object for the simulation.
    main_ifo_config : InterferometerConfig
        A configured interferometer object for the main channel.
    fitter_method : str
        The string name of the fitter to use (e.g., 'nls', 'wdfmi_ortho').
    fitter_kwargs : dict, optional
        A dictionary of keyword arguments to be passed to the `dff.fit()`
        method. This includes parameters like `n`, `ndata`, etc.
        Defaults to an empty dictionary.
    witness_ifo_config : InterferometerConfig, optional
        If provided, a witness channel is configured and simulated.
        The W-DFMI fitters require this. Defaults to None.
    n_seconds : float, optional
        The duration of the simulation in seconds. If None, it is inferred
        from the main channel's `fit_n` and `f_mod` to be a single buffer.
        Defaults to None.
    trial_num : int, optional
        An integer seed for the random number generators in the simulation,
        ensuring reproducibility. Defaults to 0.

    Returns
    -------
    DeepFitObject or None
        The fit object containing the results of the trial. Returns None if
        the simulation or fit fails.
    """
    if fitter_kwargs is None:
        fitter_kwargs = {}

    dff = dfm.DeepFitFramework()

    # --- 1. Configure Channels ---
    main_label = "main_trial"
    main_channel = dfm.DFMIObject(main_label, laser_config, main_ifo_config)
    dff.sims[main_label] = main_channel

    witness_label = None
    if witness_ifo_config:
        witness_label = "witness_trial"
        # The witness channel must share the same laser configuration
        witness_channel = dfm.DFMIObject(witness_label, laser_config, witness_ifo_config)
        dff.sims[witness_label] = witness_channel

    # --- 2. Simulate ---
    if n_seconds is None:
        # Default to a single-buffer simulation
        fit_n = fitter_kwargs.get('n', main_channel.fit_n)
        n_seconds = fit_n / laser_config.f_mod

    dff.simulate(
        main_label,
        n_seconds=n_seconds,
        witness_label=witness_label,
        trial_num=trial_num
    )

    # --- 3. Fit ---
    # For W-DFMI methods, the witness label must be passed to the fitter
    if 'wdfmi' in fitter_method:
        fitter_kwargs['witness_label'] = witness_label
        
    # Set verbose to False to avoid cluttering parallel worker output
    fitter_kwargs['verbose'] = False

    fit_obj = dff.fit(main_label, method=fitter_method, **fitter_kwargs)

    return fit_obj

def run_efficiency_trial(params: dict) -> float:
    """
    A self-contained worker for a single fitter efficiency trial.

    This worker has been refactored to use the new `experiments.run_single_trial`
    runner. Its responsibility is now limited to configuring the physics of the
    trial and specifying the fitter to be used. The runner handles all the
    boilerplate of simulation and fitting.

    Parameters
    ----------
    params : dict
        A dictionary containing all parameters for this single trial:
        - 'laser_config': A configured LaserConfig object.
        - 'ifo_config': A configured InterferometerConfig object.
        - 'n_seconds': The duration of the simulation.
        - 'ndata': The number of harmonics for the NLS fit.
        - 'm_true': The true modulation depth, used as an initial guess.
        - 'trial_num': The seed for the random number generator.

    Returns
    -------
    float
        The fitted modulation depth `m_fit` for this trial. Returns `np.nan`
        if the fit fails.
    """
    # --- 1. Unpack parameters from the input dictionary ---
    laser_config = params['laser_config']
    ifo_config = params['ifo_config']
    n_seconds = params['n_seconds']
    ndata = params['ndata']
    m_true = params['m_true']
    trial_num = params['trial_num']
    
    # --- 2. Define the fitter and its specific arguments ---
    fitter_method = 'nls'
    fitter_kwargs = {
        'n': int(laser_config.f_mod * n_seconds), # A single buffer fit
        'ndata': ndata,
        'init_m': m_true,
        'parallel': False # Important for nested workers
    }

    # --- 3. Delegate execution to the experiment runner ---
    # The runner handles all simulation and fitting boilerplate.
    fit_obj = run_single_trial(
        laser_config=laser_config,
        main_ifo_config=ifo_config,
        fitter_method=fitter_method,
        fitter_kwargs=fitter_kwargs,
        n_seconds=n_seconds,
        trial_num=trial_num
    )

    # --- 4. Extract and return the single result value ---
    if fit_obj and fit_obj.m.size > 0:
        return fit_obj.m[0]
    else:
        return np.nan
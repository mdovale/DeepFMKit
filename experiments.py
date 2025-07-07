import logging
from typing import Optional

from . import core as dfm
from .physics import LaserConfig, InterferometerConfig

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
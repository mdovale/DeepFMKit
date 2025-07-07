from . import core as dfm
from .physics import LaserConfig, InterferometerConfig

from typing import Optional
import numpy as np
from tqdm import tqdm
import multiprocessing
import os

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

def run_monte_carlo(
    worker_func,
    n_trials: int,
    static_params: dict,
    dynamic_params_generator,
    n_cores: Optional[int] = None
) -> np.ndarray:
    """
    Runs a parallel Monte Carlo simulation by repeatedly calling a worker.

    This function provides a generic framework for running Monte Carlo studies.
    It handles the boilerplate of setting up a parallel pool, creating a list
    of jobs with varying parameters, executing the jobs, and collecting the
    results.

    Parameters
    ----------
    worker_func : callable
        The worker function to be called for each trial. This function must
        accept a single dictionary of parameters.
    n_trials : int
        The total number of Monte Carlo trials to run.
    static_params : dict
        A dictionary of parameters that remain constant for all trials.
    dynamic_params_generator : callable
        A function that takes a trial index (int) and returns a dictionary
        of parameters that change for each trial. This is used to inject
        randomness (e.g., random phases).
    n_cores : int, optional
        The number of CPU cores to use for parallel execution. If None,
        defaults to all available cores.

    Returns
    -------
    np.ndarray
        A numpy array containing the results from all trials. The shape will
        depend on the return value of the worker function.
    """
    if n_cores is None:
        n_cores = os.cpu_count()

    # --- 1. Create the full list of jobs to be run ---
    all_jobs = []
    for i in range(n_trials):
        # Get the unique, randomized parameters for this specific trial
        dynamic_params = dynamic_params_generator(i)
        
        # Combine the static and dynamic parameters into one dictionary for the worker
        job_params = {**static_params, **dynamic_params}
        all_jobs.append(job_params)

    # --- 2. Run the Simulations in Parallel ---
    print(f"Starting Monte Carlo run with {len(all_jobs)} trials on {n_cores} cores...")
    
    # The 'if __name__ == "__main__"` guard is critical for scripts,
    # but since this is a library, we rely on the caller to use it.
    with multiprocessing.Pool(processes=n_cores) as pool:
        results_iterator = pool.imap(worker_func, all_jobs)
        # Use tqdm to show a progress bar for the parallel execution
        all_results = list(tqdm(results_iterator, total=len(all_jobs), desc="Monte Carlo Trials"))

    return np.array(all_results)
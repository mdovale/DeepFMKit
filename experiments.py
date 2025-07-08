# Import DeepFMKit components
from . import core as dfm
from .physics import *

import numpy as np
import itertools
import multiprocessing
import os
from tqdm import tqdm
from typing import Optional, Callable, Dict, Any, List

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
    worker_func: Callable,
    n_trials: int,
    static_params: dict,
    dynamic_params_generator: Callable,
    n_cores: Optional[int] = None,
    verbose: bool = True
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
    verbose : bool, optional
        If True, a tqdm progress bar is displayed. Defaults to True.

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
    if verbose:
        print(f"Starting Monte Carlo run with {len(all_jobs)} trials on {n_cores} cores...")
    
    with multiprocessing.Pool(processes=n_cores) as pool:
        results_iterator = pool.imap(worker_func, all_jobs)
        
        # Conditionally wrap the iterator with tqdm for a progress bar
        if verbose:
            all_results = list(tqdm(results_iterator, total=len(all_jobs), desc="Monte Carlo Trials"))
        else:
            all_results = list(results_iterator)

    return np.array(all_results)

def generic_trial_setup(params: dict) -> dict:
    """
    Configures physics objects based on a parameter dictionary and a mapping.
    
    This single, generic function replaces all previous `setup_...` functions.
    It creates the config objects, performs fixed physical calculations, and then
    dynamically sets attributes on the config objects based on the provided map.
    """
    param_map = params['__param_map__']
    
    # Create base configurations
    configs = {
        'laser_config': LaserConfig(),
        'main_ifo_config': InterferometerConfig(),
        'witness_ifo_config': InterferometerConfig()
    }

    # Perform fixed, common calculations first
    if 'm_target' in params and 'opd' in params:
        opd = params['opd']
        m_target = params['m_target']
        if opd > 0:
            configs['laser_config'].df = (m_target * dfm.sc.c) / (2 * np.pi * opd)
    
    # Dynamically set attributes using the map
    for param_name, mapping in param_map.items():
        if param_name in params:
            target_obj_name = mapping['target_object']
            target_attribute = mapping['target_attribute']
            setattr(configs[target_obj_name], target_attribute, params[param_name])
            
    return configs

def _execute_single_trial_worker(job_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a single, fully-defined simulation and analysis trial.
    This worker is completely generic and data-driven.
    """
    params = job_params['params']
    analyses = job_params['analyses']
    grid_indices = job_params['grid_indices']
    trial_idx = job_params['trial_idx']

    # Setup Physics
    configs = generic_trial_setup(params)
    laser_config, main_ifo_config, witness_ifo_config = \
        configs['laser_config'], configs['main_ifo_config'], configs.get('witness_ifo_config')

    # Simulate
    dff = dfm.DeepFitFramework()
    main_label = "main"
    main_channel = dfm.DFMIObject(main_label, laser_config, main_ifo_config)
    dff.sims[main_label] = main_channel

    witness_label = None
    if any('wdfmi' in a['fitter'] for a in analyses.values()):
        witness_label = "witness"
        witness_channel = dfm.DFMIObject(witness_label, laser_config, witness_ifo_config)
        dff.sims[witness_label] = witness_channel

    n_seconds = main_channel.fit_n / laser_config.f_mod
    dff.simulate(main_label, n_seconds=n_seconds, witness_label=witness_label)

    # Run all defined analyses
    results = {'grid_indices': grid_indices, 'trial_idx': trial_idx}
    fit_param_names = ['m', 'phi', 'psi', 'amp', 'ssq', 'dc']

    for analysis_name, analysis_def in analyses.items():
        fitter_method = analysis_def['fitter']
        fitter_kwargs = analysis_def['kwargs']
        
        if 'wdfmi' in fitter_method:
            fitter_kwargs['witness_label'] = witness_label
            
        fit_obj = dff.fit(main_label, method=fitter_method, verbose=False, **fitter_kwargs)
        
        # NEW: Automatically extract all standard fit parameters
        if fit_obj:
            analysis_results = {p: getattr(fit_obj, p)[0] for p in fit_param_names if hasattr(fit_obj, p) and getattr(fit_obj, p).size > 0}
            results[analysis_name] = analysis_results
        else:
            results[analysis_name] = {p: np.nan for p in fit_param_names}
            
    return results

class Experiment:
    """
    A declarative framework for designing and running complex simulations.
    """
    def __init__(self, description: str = ""):
        self.description = description
        self.axes: Dict[str, np.ndarray] = {}
        self.static_params: Dict[str, Any] = {}
        self.stochastic_vars: Dict[str, Callable] = {}
        self.param_map: Dict[str, Dict] = {}
        self.n_trials: int = 1
        self.analyses: Dict[str, Dict] = {}
        self.results: Optional[Dict[str, Any]] = None

    def add_axis(self, name: str, values: List[Any]):
        """Defines an independent variable for the experiment grid."""
        if len(self.axes) >= 3: raise ValueError("Supports a maximum of 3 axes.")
        self.axes[name] = np.asarray(values)

    def set_static(self, params_dict: Dict[str, Any]):
        """Sets parameters that are fixed for the entire experiment."""
        self.static_params.update(params_dict)

    def add_stochastic_variable(self, name: str, distribution_func: Callable):
        """Defines a variable to be randomized for each Monte Carlo trial."""
        self.stochastic_vars[name] = distribution_func

    def map_parameter(self, name: str, target_object: str, target_attribute: str):
        """Maps an experiment parameter to an attribute on a physics config object."""
        self.param_map[name] = {'target_object': target_object, 'target_attribute': target_attribute}

    def add_analysis(self, name: str, fitter_method: str, fitter_kwargs: Optional[dict] = None):
        """Defines an analysis (a specific fitter run) to perform on each trial."""
        self.analyses[name] = {'fitter': fitter_method, 'kwargs': fitter_kwargs or {}}
        
    def run(self, n_cores: Optional[int] = None, verbose: bool = True):
        """Executes the entire defined experiment in parallel."""
        if not self.analyses: raise ValueError("At least one analysis must be added.")
        if n_cores is None: n_cores = os.cpu_count()

        job_grid = list(itertools.product(*self.axes.values()))
        all_jobs = []
        for grid_idx, grid_point_values in enumerate(job_grid):
            for trial_idx in range(self.n_trials):
                params = dict(zip(list(self.axes.keys()), grid_point_values))
                params.update(self.static_params)
                for name, func in self.stochastic_vars.items(): params[name] = func()
                params['__param_map__'] = self.param_map
                all_jobs.append({'analyses': self.analyses, 'params': params, 'grid_indices': grid_idx, 'trial_idx': trial_idx})
        
        if verbose: print(f"Running experiment '{self.description}' with {len(all_jobs)} total trials...")

        with multiprocessing.Pool(processes=n_cores) as pool:
            results_iterator = pool.imap(_execute_single_trial_worker, all_jobs)
            if verbose: flat_results = list(tqdm(results_iterator, total=len(all_jobs), desc="Executing Trials"))
            else: flat_results = list(results_iterator)

        self.results = {'axes': self.axes}
        grid_shape = [len(v) for v in self.axes.values()]
        fit_param_names = ['m', 'phi', 'psi', 'amp', 'ssq', 'dc']
        
        for analysis_name in self.analyses:
            self.results[analysis_name] = {}
            for param_name in fit_param_names:
                analysis_grid_shape = (*grid_shape, self.n_trials)
                trial_results_grid = np.full(analysis_grid_shape, np.nan)
                
                for res in flat_results:
                    multi_dim_indices = np.unravel_index(res['grid_indices'], grid_shape)
                    full_indices = (*multi_dim_indices, res['trial_idx'])
                    if analysis_name in res and param_name in res[analysis_name]:
                        trial_results_grid[full_indices] = res[analysis_name][param_name]

                self.results[analysis_name][param_name] = {
                    'all_trials': trial_results_grid,
                    'mean': np.nanmean(trial_results_grid, axis=-1),
                    'std': np.nanstd(trial_results_grid, axis=-1),
                    'worst': np.nanmax(np.abs(trial_results_grid), axis=-1)
                }
        
        if verbose: print("Experiment finished.")
        return self.results
    
    def plot(self, analysis_name: str, param_to_plot: str, stat: str = 'mean', ax=None):
        """Visualizes 1D or 2D experiment results using Matplotlib."""
        if self.results is None: raise RuntimeError("Experiment has not been run.")
        import matplotlib.pyplot as plt
        num_axes = len(self.axes)
        if num_axes not in [1, 2]: raise ValueError(f"Plotting is only for 1D/2D experiments.")
        
        data = self.results[analysis_name][param_to_plot][stat]
        
        if ax is None: fig, ax = plt.subplots(figsize=(10, 7))
        else: fig = ax.get_figure()

        axis_names = list(self.axes.keys())
        
        if num_axes == 1:
            x_vals = self.axes[axis_names[0]]
            ax.plot(x_vals, data)
            if stat == 'mean': ax.fill_between(x_vals, data - self.results[analysis_name][param_to_plot]['std'], data + self.results[analysis_name][param_to_plot]['std'], alpha=0.2)
            ax.set_xlabel(axis_names[0])
            ax.set_ylabel(f"{stat.capitalize()} of {param_to_plot}")
        
        elif num_axes == 2:
            y_vals, x_vals = self.axes[axis_names[0]], self.axes[axis_names[1]]
            c = ax.pcolormesh(x_vals, y_vals, data, shading='gouraud')
            fig.colorbar(c, ax=ax, label=f"{stat.capitalize()} of {param_to_plot}")
            ax.set_xlabel(axis_names[1])
            ax.set_ylabel(axis_names[0])
        
        ax.set_title(f"{self.description}:\n{analysis_name} - {param_to_plot}")
        ax.grid(True, linestyle=':')
        return ax

    def plot_3d(self, analysis_name: str, stat: str = 'mean'):
        """Visualizes 3D experiment results using Plotly."""
        if self.results is None:
            raise RuntimeError("Experiment has not been run yet. Call `run()` first.")

        if len(self.axes) != 3:
            raise ValueError("3D plotting is only supported for 3D experiments.")
            
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is required for 3D plotting. Please install it: `pip install plotly`")

        data = self.results[analysis_name][stat]
        axis_names = list(self.axes.keys())
        x, y, z = np.meshgrid(*self.axes.values(), indexing='ij')

        fig = go.Figure(data=go.Scatter3d(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            mode='markers',
            marker=dict(
                size=5,
                color=data.flatten(),
                colorscale='Viridis',
                colorbar_title=f"{stat.capitalize()} of {analysis_name}",
                opacity=0.8
            )
        ))
        
        fig.update_layout(
            title=f"{self.description}: {analysis_name}",
            scene=dict(
                xaxis_title=axis_names[0],
                yaxis_title=axis_names[1],
                zaxis_title=axis_names[2]
            ),
            margin=dict(r=20, b=10, l=10, t=40)
        )
        fig.show()
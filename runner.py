import numpy as np
import itertools
import multiprocessing
import os
import copy
from tqdm import tqdm
from typing import Optional, Callable, Dict, Any, List
from . import core as dfm

def _execute_single_trial_worker(job_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a single, fully-defined simulation and analysis trial.

    This lean worker is the core computational unit for the Experiment class.
    It takes a dictionary containing all necessary functions and parameters,
    and returns a dictionary of results.

    Parameters
    ----------
    job_params : dict
        A dictionary containing all information for the trial:
        - 'trial_setup_func': The function to set up physical configs.
        - 'analyses': A dictionary defining the analyses to run.
        - 'params': A dictionary of the specific physical parameters for this trial.
        - 'grid_indices': A tuple of indices for reconstructing the results grid.

    Returns
    -------
    dict
        A dictionary containing the grid indices and the computed results for each analysis.
    """
    # Unpack the job description
    setup_func = job_params['trial_setup_func']
    analyses = job_params['analyses']
    params = job_params['params']
    grid_indices = job_params['grid_indices']

    # --- 1. Setup Physics ---
    # The user-provided function returns the configured objects
    configs = setup_func(params)
    laser_config = configs['laser_config']
    main_ifo_config = configs['main_ifo_config']
    witness_ifo_config = configs.get('witness_ifo_config') # Optional

    # --- 2. Simulate ---
    dff = dfm.DeepFitFramework()
    main_label = "main"
    main_channel = dfm.DFMIObject(main_label, laser_config, main_ifo_config)
    dff.sims[main_label] = main_channel

    witness_label = None
    if witness_ifo_config:
        witness_label = "witness"
        witness_channel = dfm.DFMIObject(witness_label, laser_config, witness_ifo_config)
        dff.sims[witness_label] = witness_channel

    n_seconds = main_channel.fit_n / laser_config.f_mod
    dff.simulate(main_label, n_seconds=n_seconds, witness_label=witness_label)

    # --- 3. Run all defined analyses ---
    results = {'grid_indices': grid_indices}
    for analysis_name, analysis_def in analyses.items():
        fitter_method = analysis_def['fitter']
        fitter_kwargs = analysis_def['kwargs']
        extractor_func = analysis_def['extractor']
        
        # Add witness label if needed by the fitter
        if 'wdfmi' in fitter_method:
            fitter_kwargs['witness_label'] = witness_label
            
        fit_obj = dff.fit(main_label, method=fitter_method, verbose=False, **fitter_kwargs)
        
        # Use the extractor to get the final scalar value
        if fit_obj:
            results[analysis_name] = extractor_func(fit_obj, params)
        else:
            results[analysis_name] = np.nan
            
    # --- THE FIX IS HERE ---
    # The runner needs the original parameters back to know the trial index.
    results['params'] = params
            
    return results

class Experiment:
    """
    A declarative framework for designing and running complex simulations.

    This class provides a high-level API to define a parameter space,
    stochastic variables, the physics of a single trial, and the analyses to
    be performed. It abstracts away the boilerplate of looping, parallelization,
    and result aggregation.

    Example
    -------
    >>> exp = Experiment("My first experiment")
    >>> exp.add_axis('m_main', np.linspace(5, 10, 3))
    >>> exp.add_stochastic_variable('phase', lambda: np.random.rand())
    >>> exp.set_trial_setup(my_setup_function)
    >>> exp.add_analysis('bias', 'nls', my_extractor_function)
    >>> exp.run()
    >>> exp.plot('bias')
    """
    def __init__(self, description: str = ""):
        self.description = description
        self.axes: Dict[str, np.ndarray] = {}
        self.static_params: Dict[str, Any] = {}
        self.stochastic_vars: Dict[str, Callable] = {}
        self.n_trials: int = 1
        self.trial_setup_func: Optional[Callable] = None
        self.analyses: Dict[str, Dict] = {}
        self.results: Optional[Dict[str, Any]] = None

    def add_axis(self, name: str, values: List[Any]):
        """Defines an independent variable for the experiment grid.
        
        Supports up to 3 axes for 1D, 2D, or 3D parameter sweeps.
        
        Parameters
        ----------
        name : str
            The name of the parameter (e.g., 'm_main').
        values : list or np.ndarray
            The list of values to sweep over for this parameter.
        """
        if len(self.axes) >= 3:
            raise ValueError("Experiment class supports a maximum of 3 axes.")
        self.axes[name] = np.asarray(values)

    def set_static(self, params_dict: Dict[str, Any]):
        """Sets parameters that are fixed for the entire experiment."""
        self.static_params.update(params_dict)

    def add_stochastic_variable(self, name: str, distribution_func: Callable):
        """Defines a variable to be randomized for each Monte Carlo trial."""
        self.stochastic_vars[name] = distribution_func

    def set_trial_setup(self, func: Callable):
        """Sets the function that configures the physics for a single trial.
        
        This function must accept a single dictionary of parameters and return
        a dictionary containing the configured 'laser_config', 
        'main_ifo_config', and optional 'witness_ifo_config' objects.
        """
        self.trial_setup_func = func

    def add_analysis(self, name: str, fitter_method: str, 
                     result_extractor_func: Callable, fitter_kwargs: Optional[dict] = None):
        """Defines an analysis to run on each trial's simulated data.
        
        Parameters
        ----------
        name : str
            A unique name for this analysis (e.g., 'wdfmi_bias').
        fitter_method : str
            The string name of the fitter to use (e.g., 'nls', 'wdfmi_ortho').
        result_extractor_func : callable
            A function that takes the resulting DeepFitObject and the trial's
            parameter dictionary and returns a single scalar value.
        fitter_kwargs : dict, optional
            A dictionary of kwargs to pass to the dff.fit() method.
        """
        self.analyses[name] = {
            'fitter': fitter_method,
            'extractor': result_extractor_func,
            'kwargs': fitter_kwargs or {}
        }
        
    def run(self, n_cores: Optional[int] = None, verbose: bool = True):
        """Executes the entire defined experiment in parallel.
        
        This method builds the complete job list, distributes the trials across
        multiple CPU cores, and aggregates the results into a structured
        dictionary.
        """
        if self.trial_setup_func is None:
            raise ValueError("Trial setup function must be set using `set_trial_setup`.")
        if not self.analyses:
            raise ValueError("At least one analysis must be added using `add_analysis`.")

        if n_cores is None:
            n_cores = os.cpu_count()

        # --- 1. Build the Job Grid ---
        axis_names = list(self.axes.keys())
        axis_values = list(self.axes.values())
        job_grid = list(itertools.product(*axis_values))
        
        all_jobs = []
        # Loop over each point in the N-dimensional parameter grid
        for grid_idx, grid_point_values in enumerate(job_grid):
            # For each grid point, run `n_trials` Monte Carlo simulations
            for trial_idx in range(self.n_trials):
                # Combine axis parameters
                params = dict(zip(axis_names, grid_point_values))
                # Add static parameters
                params.update(self.static_params)
                # Add randomized stochastic parameters
                for name, func in self.stochastic_vars.items():
                    params[name] = func()
                
                # --- THE FIX IS HERE ---
                # The trial index must be part of the core `params` dictionary
                # so it is available to the worker and returned correctly.
                params['trial_idx'] = trial_idx
                
                job_params = {
                    'trial_setup_func': self.trial_setup_func,
                    'analyses': self.analyses,
                    'params': params,
                    'grid_indices': grid_idx, # Use the flattened grid index
                }
                all_jobs.append(job_params)
        
        # --- 2. Execute in Parallel ---
        if verbose:
            print(f"Running experiment '{self.description}' with {len(all_jobs)} total trials on {n_cores} cores...")

        with multiprocessing.Pool(processes=n_cores) as pool:
            results_iterator = pool.imap(_execute_single_trial_worker, all_jobs)
            if verbose:
                flat_results = list(tqdm(results_iterator, total=len(all_jobs), desc="Executing Trials"))
            else:
                flat_results = list(results_iterator)

        # --- 3. Aggregate and Reshape Results ---
        self.results = {'axes': self.axes}
        grid_shape = [len(v) for v in self.axes.values()]
        
        for analysis_name in self.analyses:
            # Create a large array to hold all trial results for this analysis
            analysis_grid_shape = (*grid_shape, self.n_trials)
            trial_results_grid = np.full(analysis_grid_shape, np.nan)
            
            # Populate the grid using the indices from the worker
            for res in flat_results:
                # The grid_indices is a flat index from itertools.product.
                # We need to convert it to a multidimensional index.
                multi_dim_indices = np.unravel_index(res['grid_indices'], grid_shape)
                trial_idx = res['params']['trial_idx'] # This will now work
                full_indices = (*multi_dim_indices, trial_idx)
                trial_results_grid[full_indices] = res[analysis_name]

            # Calculate statistics over the trials axis
            self.results[analysis_name] = {
                'all_trials': trial_results_grid,
                'mean': np.nanmean(trial_results_grid, axis=-1),
                'std': np.nanstd(trial_results_grid, axis=-1),
                'worst': np.nanmax(np.abs(trial_results_grid), axis=-1)
            }
        
        if verbose:
            print("Experiment finished.")
        return self.results
    
    def plot(self, analysis_name: str, stat: str = 'mean', ax=None):
        """Visualizes 1D or 2D experiment results using Matplotlib."""
        if self.results is None:
            raise RuntimeError("Experiment has not been run yet. Call `run()` first.")
        
        import matplotlib.pyplot as plt

        num_axes = len(self.axes)
        if num_axes not in [1, 2]:
            raise ValueError(f"Matplotlib plotting is only supported for 1D and 2D experiments. This experiment has {num_axes} axes.")

        if analysis_name not in self.results:
            raise KeyError(f"Analysis '{analysis_name}' not found in results.")
        
        data = self.results[analysis_name][stat]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        else:
            fig = ax.get_figure()

        axis_names = list(self.axes.keys())
        
        if num_axes == 1:
            x_vals = self.axes[axis_names[0]]
            ax.plot(x_vals, data)
            if stat == 'mean':
                std_data = self.results[analysis_name]['std']
                ax.fill_between(x_vals, data - std_data, data + std_data, alpha=0.2)
            ax.set_xlabel(axis_names[0])
            ax.set_ylabel(f"{stat.capitalize()} of {analysis_name}")
        
        elif num_axes == 2:
            x_vals, y_vals = self.axes[axis_names[1]], self.axes[axis_names[0]]
            # Data grid needs to be transposed for pcolormesh
            c = ax.pcolormesh(x_vals, y_vals, data, shading='gouraud')
            fig.colorbar(c, ax=ax, label=f"{stat.capitalize()} of {analysis_name}")
            ax.set_xlabel(axis_names[1])
            ax.set_ylabel(axis_names[0])
        
        ax.set_title(f"{self.description}: {analysis_name}")
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
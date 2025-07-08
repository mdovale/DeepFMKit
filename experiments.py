from DeepFMKit import core as dfm
from .physics import *

import numpy as np
import itertools
import multiprocessing
import os
from tqdm import tqdm
from typing import Optional, Callable, Dict, Any, List

def _execute_single_trial_worker(job_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a single simulation trial with pre-configured physics objects.
    """
    # Unpack the job description
    laser_config = job_params['laser_config']
    main_ifo_config = job_params['main_ifo_config']
    witness_ifo_config = job_params.get('witness_ifo_config') # Optional
    
    analyses = job_params['analyses']
    grid_indices = job_params['grid_indices']
    trial_idx = job_params['trial_idx']

    # --- 1. Simulate (no setup needed here) ---
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

    # --- 2. Run all defined analyses ---
    results = {'grid_indices': grid_indices, 'trial_idx': trial_idx}
    fit_param_names = ['m', 'phi', 'psi', 'amp', 'ssq', 'dc']

    for analysis_name, analysis_def in analyses.items():
        fitter_method = analysis_def['fitter']
        fitter_kwargs = analysis_def['kwargs']
        
        if 'wdfmi' in fitter_method:
            fitter_kwargs['witness_label'] = witness_label
            
        fit_obj = dff.fit(main_label, method=fitter_method, verbose=False, **fitter_kwargs)
        
        if fit_obj:
            analysis_results = {p: getattr(fit_obj, p)[0] for p in fit_param_names if hasattr(fit_obj, p) and getattr(fit_obj, p).size > 0}
            results[analysis_name] = analysis_results
        else:
            results[analysis_name] = {p: np.nan for p in fit_param_names}
            
    return results

class Experiment:
    """
    A declarative framework for designing and running complex simulations.
    This version uses a user-provided factory function to configure the physics.
    """
    def __init__(self, description: str = ""):
        self.description = description
        self.axes: Dict[str, np.ndarray] = {}
        self.static_params: Dict[str, Any] = {}
        self.stochastic_vars: Dict[str, Callable] = {}
        self.n_trials: int = 1
        self.analyses: Dict[str, Dict] = {}
        self.config_factory: Optional[Callable] = None
        self.results: Optional[Dict[str, Any]] = None

    def add_axis(self, name: str, values: List[Any]):
        """Defines an independent variable for the experiment grid."""
        self.axes[name] = np.asarray(values)

    def set_static(self, params_dict: Dict[str, Any]):
        """Sets parameters that are fixed for the entire experiment."""
        self.static_params.update(params_dict)

    def add_stochastic_variable(self, name: str, distribution_func: Callable):
        """Defines a variable to be randomized for each Monte Carlo trial."""
        self.stochastic_vars[name] = distribution_func
        
    def set_config_factory(self, func: Callable):
        """
        Sets the user-defined function that creates the physics configs.

        This function must accept a single dictionary of parameters for one
        trial and return a dictionary containing the configured
        'laser_config' and 'main_ifo_config' objects.
        """
        self.config_factory = func

    def add_analysis(self, name: str, fitter_method: str, fitter_kwargs: Optional[dict] = None):
        """Defines an analysis to perform on each trial."""
        self.analyses[name] = {'fitter': fitter_method, 'kwargs': fitter_kwargs or {}}
        
    def run(self, n_cores: Optional[int] = None, verbose: bool = True):
        """Executes the entire defined experiment in parallel."""
        if self.config_factory is None:
            raise ValueError("A configuration factory must be set using .set_config_factory()")
        if not self.analyses:
            raise ValueError("At least one analysis must be added using .add_analysis()")
        if n_cores is None:
            n_cores = os.cpu_count()

        job_grid = list(itertools.product(*self.axes.values()))
        all_jobs = []
        for grid_idx, grid_point_values in enumerate(job_grid):
            for trial_idx in range(self.n_trials):
                # Build the complete parameter set for this trial
                params = dict(zip(list(self.axes.keys()), grid_point_values))
                params.update(self.static_params)
                for name, func in self.stochastic_vars.items():
                    params[name] = func()
                
                # Use the user's factory to get the configured physics objects
                configs = self.config_factory(params)

                all_jobs.append({
                    **configs, # Unpack laser_config, main_ifo_config, etc.
                    'analyses': self.analyses,
                    'grid_indices': grid_idx,
                    'trial_idx': trial_idx
                })
        
        if verbose: print(f"Running experiment '{self.description}' with {len(all_jobs)} total trials...")
        
        with multiprocessing.Pool(processes=n_cores) as pool:
            results_iterator = pool.imap(_execute_single_trial_worker, all_jobs)
            if verbose: flat_results = list(tqdm(results_iterator, total=len(all_jobs), desc="Executing Trials"))
            else: flat_results = list(results_iterator)

        # Result aggregation logic remains the same
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
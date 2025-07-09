from DeepFMKit import core as dfm
from .physics import *
from .factories import ExperimentFactory

import numpy as np
import itertools
import multiprocessing
import os
import copy
from tqdm import tqdm
from typing import Optional, Callable, Dict, Any, List, Union

def _run_single_trial(job_packet: tuple) -> dict:
    """
    Worker function for parallel processing. Executes one full trial.
    (This function remains unchanged from our previous refactor)
    """
    # 1. Unpack the job packet
    trial_params, config_factory, analyses_to_run, n_seconds, f_samp, trial_num = job_packet

    # 2. Create the physics configurations for this trial
    configs = config_factory(trial_params)

    # 3. Instantiate and run the simulation using a local DFF object
    dff_local = dfm.DeepFitFramework()
    main_sim_label = "main"
    main_channel_sim = DFMIObject(label=main_sim_label, laser_config=configs['laser_config'],
                                  ifo_config=configs['main_ifo_config'], f_samp=f_samp)
    dff_local.sims[main_sim_label] = main_channel_sim

    witness_sim_label = None
    if 'witness_ifo_config' in configs:
        witness_sim_label = "witness"
        witness_channel_sim = DFMIObject(label=witness_sim_label, laser_config=configs['laser_config'],
                                         ifo_config=configs['witness_ifo_config'], f_samp=f_samp)
        dff_local.sims[witness_sim_label] = witness_channel_sim

    dff_local.simulate(
        main_label=main_sim_label, 
        n_seconds=n_seconds, 
        mode='asd',
        trial_num=trial_num,
        witness_label=witness_sim_label
    )

    # 4. Run the requested analyses
    trial_results = {}
    for analysis in analyses_to_run:
        analysis_name = analysis['name']
        fitter_args = copy.deepcopy(analysis.get('fitter_kwargs', {}))
        fitter_args.update({
            'method': analysis['fitter_method'],
            'main_label': main_sim_label
        })
        if analysis['fitter_method'] in ['nls', 'ekf']:
            fitter_args['parallel'] = False
        if 'wdfmi' in analysis['fitter_method'] or 'hwdfmi' in analysis['fitter_method']:
            fitter_args['witness_label'] = witness_sim_label

        fit_obj = dff_local.fit(**fitter_args)
        
        # 5. Extract and store the results for this analysis
        if fit_obj:
            results_df = dff_local.fits_df[fit_obj.label]
            mean_results = results_df.mean().to_dict()
            trial_results[analysis_name] = mean_results
        else:
            trial_results[analysis_name] = {}
            
    # 6. Return the results packet
    return {'point_params': trial_params, 'results': trial_results}


class Experiment:
    """
    A declarative framework for defining and running complex simulation experiments.
    """
    def __init__(self, description: str = "Unnamed Experiment"):
        self.description = description
        self.axes: Dict[str, np.ndarray] = {}
        self.static_params: Dict[str, Any] = {}
        self.stochastic_vars: Dict[str, Dict[str, Any]] = {}
        self.config_factory: Optional[ExperimentFactory] = None
        self.analyses: List[Dict[str, Any]] = []
        self.n_trials: int = 1
        self.n_seconds_per_trial: float = 1.0
        self.f_samp: int = 200000

    def add_axis(self, name: str, values: np.ndarray):
        """Adds a new parameter axis to the experiment's sweep space."""
        self.axes[name] = np.asarray(values)

    def set_static(self, params: Dict[str, Any]):
        """Sets parameters that remain constant across all trials and axes."""
        self.static_params.update(params)

    def add_stochastic_variable(self, name: str, generator_func: Callable, depends_on: Optional[str] = None):
        """Adds a stochastic variable for Monte Carlo trials."""
        self.stochastic_vars[name] = {'generator': generator_func, 'depends_on': depends_on}

    def set_config_factory(self, factory: ExperimentFactory):
        """
        Sets the factory object that generates physics configurations.
        The factory must be an instance of a class that inherits from
        ExperimentFactory.
        """
        if not isinstance(factory, ExperimentFactory):
            raise TypeError("factory must be an instance of a class that inherits from ExperimentFactory.")
        self.config_factory = factory

    def add_analysis(self, name: str, fitter_method: str, result_cols: Optional[List[str]] = None, fitter_kwargs: Optional[Dict[str, Any]] = None):
        """Defines an analysis to be run on the simulated data."""
        self.analyses.append({
            'name': name,
            'fitter_method': fitter_method,
            'result_cols': result_cols,
            'fitter_kwargs': fitter_kwargs or {}
        })

    def get_params_for_point(self, axis_idx: Union[int, tuple]) -> Dict[str, Any]:
        """
        Retrieves a representative dictionary of parameters for a specific point
        on the experiment's N-dimensional axis grid.

        This is a crucial utility for post-processing, allowing for the
        re-creation of the exact non-stochastic parameters used at a specific
        grid point to calculate ground-truth values.

        For any stochastic variables defined in the experiment, this method will
        generate a deterministic, representative value by temporarily setting a
        fixed seed for the random number generator. This ensures that repeated
        calls for the same `axis_idx` yield the same parameter set.

        Parameters
        ----------
        axis_idx : int or tuple
            The index (or tuple of indices for multi-axis experiments) for the
            specific point on the grid. For a 1D experiment, an integer is
            sufficient. For a 2D experiment with axes ('A', 'B'), an index
            like (3, 5) would correspond to the 4th value of axis 'A' and the
            6th value of axis 'B'.

        Returns
        -------
        dict
            A dictionary containing all parameter values (static, axis-dependent,
            and a deterministically-generated value for each stochastic variable)
            for the specified grid point.
            
        Raises
        ------
        ValueError
            If the dimension of `axis_idx` does not match the number of defined axes.
        """
        # 1. Start with a deep copy of the static parameters.
        # A deep copy is used to prevent any accidental modification of the
        # original experiment's state.
        params = copy.deepcopy(self.static_params)
        
        # 2. Add the axis-dependent parameters for the specified grid point.
        axis_names = list(self.axes.keys())
        
        # For user convenience, gracefully handle a single integer index for 1D experiments.
        if isinstance(axis_idx, int):
            # Convert the integer to a single-element tuple to make the logic consistent.
            axis_idx = (axis_idx,)

        # Check that the provided index dimension matches the number of axes.
        if len(axis_idx) != len(axis_names):
            raise ValueError(
                f"Dimension of axis_idx ({len(axis_idx)}) does not match the number "
                f"of defined axes ({len(axis_names)})."
            )

        # Loop through the axes and add the corresponding value for the given index.
        for i, axis_name in enumerate(axis_names):
            params[axis_name] = self.axes[axis_name][axis_idx[i]]
            
        # 3. Add representative values for all stochastic variables.
        # To ensure this function is deterministic, we temporarily fix the random
        # state, generate the values, and then restore the original state.
        original_random_state = np.random.get_state()
        np.random.seed(0) # Use a fixed seed for reproducible "random" values.

        for var_name, var_info in self.stochastic_vars.items():
            generator = var_info['generator']
            dependency_name = var_info.get('depends_on')
            
            if dependency_name:
                # If the generator depends on an axis value, pass it.
                if dependency_name not in params:
                    # This is a safeguard against configuration errors.
                    np.random.set_state(original_random_state) # Restore state before raising
                    raise ValueError(
                        f"Stochastic variable '{var_name}' depends on "
                        f"'{dependency_name}', which is not a defined axis or static parameter."
                    )
                params[var_name] = generator(params[dependency_name])
            else:
                # Otherwise, call the generator without arguments.
                params[var_name] = generator()
        
        # IMPORTANT: Restore the global random state to not affect other parts of the user's script.
        np.random.set_state(original_random_state)
            
        # 4. Return the complete, representative parameter set.
        return params

    def run(self, n_cores: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes the defined experiment, parallelizing over all individual trials.

        This method orchestrates the entire experiment by first generating a
        complete list of all parameter combinations (for all axes and all trials).
        It then distributes these "jobs" to a pool of worker processes.
        Finally, it collects and aggregates the results from all workers into a
        structured dictionary.

        Parameters
        ----------
        n_cores : int, optional
            The number of CPU cores to use for parallelizing the experiment.
            If None, uses all available cores as reported by os.cpu_count().

        Returns
        -------
        dict
            A nested dictionary containing the aggregated results.
            Structure: results[analysis_name][parameter_name]['mean'/'std'/'all_trials']
        """
        # --- 0. Pre-flight Checks ---
        if self.config_factory is None:
            raise ValueError("A configuration factory must be set using set_config_factory().")
        if not self.axes:
            raise ValueError("At least one parameter axis must be defined using add_axis().")
        
        if n_cores is None:
            n_cores = os.cpu_count()

        # --- 1. Generate the Full List of All Jobs ---
        job_packets = []
        
        # Get the names of all defined axes
        axis_names = list(self.axes.keys())
        
        # Create a list of index ranges for each axis
        axis_indices_list = [range(len(ax)) for ax in self.axes.values()]
        
        # Use itertools.product to create every possible combination of indices.
        # For a 2D sweep of size (50, 20), this creates tuples like (0,0), (0,1), ..., (49,19).
        axis_combinations_indices = list(itertools.product(*axis_indices_list))

        trial_counter = 0
        # Loop through each point on the N-dimensional parameter grid
        for point_indices_tuple in axis_combinations_indices:
            # Start with static parameters for this point
            point_params = copy.deepcopy(self.static_params)
            
            # Add the specific axis values for this grid point
            for i, axis_name in enumerate(axis_names):
                point_params[axis_name] = self.axes[axis_name][point_indices_tuple[i]]
            
            # Now, create a job for each Monte Carlo trial at this grid point
            for j in range(self.n_trials):
                trial_params = copy.deepcopy(point_params)
                
                # Add the special index keys needed for result aggregation later.
                # This is crucial for correctly reassembling the results.
                trial_params['_exp_point_idx'] = point_indices_tuple
                trial_params['_exp_trial_idx'] = j
                
                # Generate and add the stochastic variables for this specific trial
                for var_name, var_info in self.stochastic_vars.items():
                    generator = var_info['generator']
                    dependency_name = var_info.get('depends_on')
                    if dependency_name:
                        trial_params[var_name] = generator(trial_params[dependency_name])
                    else:
                        trial_params[var_name] = generator()
                
                # Create the final job "packet" (a simple tuple) and add it to the list.
                # This packet is guaranteed to be pickleable.
                job_packets.append((
                    trial_params, self.config_factory, self.analyses,
                    self.n_seconds_per_trial, self.f_samp, trial_counter
                ))
                trial_counter += 1

        logging.info(f"Starting experiment '{self.description}' with {len(job_packets)} total trials.")
        logging.info(f"Using {n_cores} CPU cores for parallel processing.")
        
        # --- 2. Execute All Jobs in Parallel ---
        # The 'with' statement ensures the pool is properly closed.
        with multiprocessing.Pool(processes=n_cores) as pool:
            # pool.imap is an iterator that yields results as they complete.
            # tqdm wraps this iterator to create a live progress bar.
            flat_results = list(tqdm(pool.imap(_run_single_trial, job_packets), total=len(job_packets), desc="Running Trials"))
            
        # --- 3. Aggregate the Results ---
        # Initialize the top-level results dictionary
        results = {'axes': self.axes}
        
        # Determine the columns to collect for each analysis
        for analysis in self.analyses:
            analysis_name = analysis['name']
            results[analysis_name] = {}
            cols_to_collect = analysis.get('result_cols')
            
            # If the user didn't specify columns, we intelligently find all
            # unique result keys produced by that analysis.
            if cols_to_collect is None:
                all_keys = set()
                for res_packet in flat_results:
                    if analysis_name in res_packet['results']:
                        all_keys.update(res_packet['results'][analysis_name].keys())
                cols_to_collect = sorted(list(all_keys))

            # Initialize the final numerical grid for each parameter with NaNs.
            # The shape is (len(axis1), len(axis2), ..., n_trials).
            for col in cols_to_collect:
                shape = tuple(len(ax) for ax in self.axes.values()) + (self.n_trials,)
                all_trials_grid = np.full(shape, np.nan, dtype=float)
                results[analysis_name][col] = {'all_trials': all_trials_grid}

        # Populate the final grids directly from the flat list of results
        for packet in flat_results:
            # Retrieve the N-dimensional index and trial index from the packet
            point_idx = packet['point_params']['_exp_point_idx']
            trial_idx = packet['point_params']['_exp_trial_idx']
            # Combine them to get the full index for the final grid
            full_idx = point_idx + (trial_idx,)
            
            for analysis in self.analyses:
                analysis_name = analysis['name']
                if analysis_name in packet['results']:
                    # Place the result for each parameter into its correct grid location
                    for col, stats_dict in results[analysis_name].items():
                        val = packet['results'][analysis_name].get(col, np.nan)
                        stats_dict['all_trials'][full_idx] = val
        
        # Calculate summary statistics (mean, std) across the trial dimension
        for analysis_name, res_dict in results.items():
            if analysis_name == 'axes': continue
            for col, stats_dict in res_dict.items():
                stats_dict['mean'] = np.nanmean(stats_dict['all_trials'], axis=-1)
                stats_dict['std'] = np.nanstd(stats_dict['all_trials'], axis=-1)

        logging.info("Experiment run complete. Results aggregated.")
        return results
    
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
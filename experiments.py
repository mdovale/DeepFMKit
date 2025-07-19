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
import pickle


def _run_single_trial(job_packet: tuple) -> dict:
    """
    Worker function for parallel processing. Executes one full trial.
    Configure -> Simulate -> Analyze.
    """
    # Unpack the job packet
    trial_params, config_factory, analyses_to_run, num_fit_buffers, f_samp, trial_num = job_packet

    # Create the physics configurations for this trial
    configs = config_factory(trial_params)
    laser_config=configs['laser_config']

    # Calculate R (raw samples per fit buffer) based on the sampling frequency
    R = int(f_samp / laser_config.f_mod)

    # Calculate the total number of raw samples needed for this trial.
    num_samples_needed = num_fit_buffers * R
    if num_samples_needed == 0:
        # Handle edge case where calculated samples needed is zero
        logging.warning("Calculated num_samples_needed is zero. Setting to 1 to avoid division by zero.")
        num_samples_needed = 1

    # Calculate the actual simulation time in seconds.
    n_seconds_to_simulate = num_samples_needed / f_samp

    # Instantiate and run the simulation using a local DFF object
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
        n_seconds=n_seconds_to_simulate, 
        mode='asd',
        trial_num=trial_num,
        witness_label=witness_sim_label
    )

    # Run the requested analyses
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

        fitter_args['n'] = num_fit_buffers
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

    The Experiment class is designed for high-throughput parallel execution. To avoid 
    critical multiprocessing errors, the following rules must be strictly followed:

    - The Worker is the Atomic Unit: The top-level worker function (_run_single_trial) 
    is the atomic unit of work and encapsulates the entire "simulate-then-fit" pipeline 
    for one set of parameters.

    - No Nested Parallelism: Fitters called by a parallel Experiment run must be executed 
    in sequential mode (e.g., StandardNLSFitter with parallel=False). The Experiment class
    is the sole manager of the multiprocessing.Pool.

    - User Logic Must Be Pickleable: All user-defined logic for configuring an experiment 
    (e.g., creating physics objects, defining custom waveforms) must be contained within 
    a class that inherits from ExperimentFactory. The user passes an instance of this class
    to the experiment. This pattern ensures all necessary code and data can be safely 
    "pickled" and sent to worker processes, avoiding AttributeError on __main__.
    """
    def __init__(self, description: str = "Unnamed Experiment", filename: Optional[str] = None):
        self.description = description
        self.axes: Dict[str, np.ndarray] = {}
        self.static_params: Dict[str, Any] = {}
        self.stochastic_vars: Dict[str, Dict[str, Any]] = {}
        self.config_factory: Optional[ExperimentFactory] = None
        self._expected_params_keys: Set[str] = set() # Store expected parameters for validation
        self.analyses: List[Dict[str, Any]] = []
        self.n_trials: int = 1
        self.n_fit_buffers_per_trial: int = 10
        self.f_samp: int = 200000
        self.results: Optional[Dict[str, Any]] = None # To store aggregated results after run()

        if filename is not None:
            self.load_results(filename)

    def _validate_param_name(self, name: str):
        """Internal helper to validate if a parameter name is expected by the factory."""
        if not self._expected_params_keys:
            logging.warning("No config factory set yet. Parameter validation will be skipped until set_config_factory() is called.")
            return

        if name not in self._expected_params_keys:
            raise ValueError(
                f"Parameter '{name}' is not recognized by the current "
                f"ExperimentFactory ({type(self.config_factory).__name__}).\n"
                f"Expected parameters are: {sorted(list(self._expected_params_keys))}.\n"
                f"Please update your ExperimentFactory to handle this parameter "
                f"or remove it from your experiment configuration."
            )

    def add_axis(self, name: str, values: np.ndarray):
        """Adds a new parameter axis to the experiment's sweep space."""
        self._validate_param_name(name)
        self.axes[name] = np.asarray(values)

    def set_static(self, params: Dict[str, Any]):
        """Sets parameters that remain constant across all trials and axes."""
        for name in params.keys():
            self._validate_param_name(name)
        self.static_params.update(params)

    def add_stochastic_variable(self, name: str, generator_func: Callable, depends_on: Optional[str] = None):
        """Adds a stochastic variable for Monte Carlo trials."""
        self._validate_param_name(name)
        if depends_on is not None:
            self._validate_param_name(depends_on) # Also validate the dependency
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
        # Once the factory is set, retrieve its expected parameters for validation
        self._expected_params_keys = self.config_factory._get_expected_params_keys()
        logging.info(f"Config factory '{type(factory).__name__}' set. Expected parameters: {sorted(list(self._expected_params_keys))}")


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
            
        # 4. Filter parameters to include only those expected by the factory,
        # plus the internal _exp_point_idx and _exp_trial_idx
        filtered_params = {k: v for k, v in params.items() if k in self._expected_params_keys or k.startswith('_exp_')}
        return filtered_params
    
    def save_results(self, filename: str):
        """Save current experiment results to disk."""
        if self.results is None:
            raise RuntimeError("No results to save. Run the experiment first.")
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
        logging.info(f"Experiment results saved to {filename}")

    def load_results(self, filename: str):
        """Load experiment results from disk into the object."""
        with open(filename, 'rb') as f:
            self.results = pickle.load(f)
        logging.info(f"Experiment results loaded from {filename}")

    def run(self, n_cores: Optional[int] = None, filename: Optional[str] = None) -> Dict[str, Any]:
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
        if not self.axes and not self.n_trials > 0:
             raise ValueError("At least one parameter axis must be defined using add_axis(), or n_trials must be > 0.")
        
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
                
                # Filter trial_params to include only those keys that the factory expects
                # along with internal experiment indices. This prevents passing unexpected
                # parameters to the factory, which might cause errors or confusion.
                trial_params_for_factory = {
                    k: v for k, v in trial_params.items()
                    if k in self._expected_params_keys or k.startswith('_exp_')
                }

                # Create the final job "packet" (a simple tuple) and add it to the list.
                # This packet is guaranteed to be pickleable.
                job_packets.append((
                    trial_params_for_factory, self.config_factory, self.analyses,
                    self.n_fit_buffers_per_trial, self.f_samp, trial_counter
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
            # Handle the case where there are no axes defined but n_trials > 0
            if not axis_names:
                shape = (self.n_trials,)
            else:
                shape = tuple(len(ax) for ax in self.axes.values()) + (self.n_trials,)
            
            for col in cols_to_collect:
                all_trials_grid = np.full(shape, np.nan, dtype=float)
                results[analysis_name][col] = {'all_trials': all_trials_grid}

        # Populate the final grids directly from the flat list of results
        for packet in flat_results:
            # Retrieve the N-dimensional index and trial index from the packet
            point_idx = packet['point_params']['_exp_point_idx']
            trial_idx = packet['point_params']['_exp_trial_idx']
            # Combine them to get the full index for the final grid
            if not axis_names: # Handle case of no axes, just trials
                full_idx = (trial_idx,)
            else:
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
                stats_dict['min'] = np.nanmin(stats_dict['all_trials'], axis=-1)
                stats_dict['max'] = np.nanmax(stats_dict['all_trials'], axis=-1)

                # Define "worst-case" as the most extreme deviation from the mean
                deviation = np.abs(stats_dict['all_trials'] - stats_dict['mean'][..., np.newaxis])
                worst_indices = np.nanargmax(deviation, axis=-1)
                worst_case = np.take_along_axis(stats_dict['all_trials'], worst_indices[..., np.newaxis], axis=-1).squeeze(-1)
                stats_dict['worst'] = worst_case

        logging.info("Experiment run complete. Results aggregated.")
        self.results = results  # Store results for potential plotting/inspection

        if filename is not None:
            self.save_results(filename)
            logging.info(f"Results saved to: {filename}")

        return results
    
    def plot(self, analysis_name: str, param_to_plot: str, stat: str = 'mean', ax=None):
        """Visualizes 1D or 2D experiment results using Matplotlib."""
        if self.results is None: raise RuntimeError("Experiment has not been run.")
        import matplotlib.pyplot as plt
        
        num_axes = len(self.axes)
        if num_axes not in [1, 2]: 
            # Handle the case where no axes are defined (n_trials only)
            if num_axes == 0 and self.n_trials > 0:
                logging.warning("No axes defined. Plotting not supported for n_trials only via this method.")
                return None
            raise ValueError(f"Plotting is only for 1D/2D experiments. This experiment has {num_axes} axes.")
        
        data = self.results[analysis_name][param_to_plot][stat]
        
        if ax is None: fig, ax = plt.subplots(figsize=(10, 7))
        else: fig = ax.get_figure()

        axis_names = list(self.axes.keys())
        
        if num_axes == 1:
            x_vals = self.axes[axis_names[0]]
            ax.plot(x_vals, data)
            if stat == 'mean': 
                # Ensure std is available before plotting fill_between
                if 'std' in self.results[analysis_name][param_to_plot]:
                    ax.fill_between(x_vals, data - self.results[analysis_name][param_to_plot]['std'], data + self.results[analysis_name][param_to_plot]['std'], alpha=0.2)
            ax.set_xlabel(axis_names[0])
            ax.set_ylabel(f"{stat.capitalize()} of {param_to_plot}")
        
        elif num_axes == 2:
            # Need to decide which axis goes on X and which on Y for pcolormesh
            # Assuming axis_names[0] is Y and axis_names[1] is X for convention with meshgrid.
            y_vals, x_vals = self.axes[axis_names[0]], self.axes[axis_names[1]]
            # Ensure data shape matches (len(y_vals), len(x_vals)) for pcolormesh
            if data.shape != (len(y_vals), len(x_vals)):
                logging.warning(f"Data shape {data.shape} does not match expected 2D axis shape ({len(y_vals)}, {len(x_vals)}). Check data processing.")
            c = ax.pcolormesh(x_vals, y_vals, data, shading='gouraud')
            fig.colorbar(c, ax=ax, label=f"{stat.capitalize()} of {param_to_plot}")
            ax.set_xlabel(axis_names[1])
            ax.set_ylabel(axis_names[0])
        
        ax.set_title(f"{self.description}:\n{analysis_name} - {param_to_plot}")
        ax.grid(True, linestyle=':')
        plt.tight_layout() # Ensure layout is tight for all plots
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
        # Meshgrid with 'ij' indexing matches numpy's array indexing (row, col, depth)
        x_mesh, y_mesh, z_mesh = np.meshgrid(*self.axes.values(), indexing='ij')

        fig = go.Figure(data=go.Scatter3d(
            x=x_mesh.flatten(),
            y=y_mesh.flatten(),
            z=z_mesh.flatten(),
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
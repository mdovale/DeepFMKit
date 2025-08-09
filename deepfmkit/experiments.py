# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
"""A declarative framework for defining and running simulation experiments.

This module provides the `Experiment` class, a high-level tool for orchestrating
large-scale, parallelized simulation studies. It is designed to handle
multi-dimensional parameter sweeps and Monte Carlo analyses in a robust and
reproducible manner.

The framework is built around key design patterns to ensure safe and efficient
parallel execution, which is crucial for high-throughput studies:
- **Atomicity**: The top-level worker function, `_run_single_trial`, is the
  atomic unit of work. It encapsulates the entire "simulate-then-analyze"
  pipeline for one unique set of parameters, ensuring that each parallel task
  is independent.
- **Serialization Safety**: All user-defined logic for configuring an experiment
  (e.g., creating physics objects, defining custom waveforms) must be
  contained within a class that inherits from `ExperimentFactory`. This
  ensures the logic is "pickleable" and can be safely transmitted to worker
  processes, avoiding common `multiprocessing` errors in interactive
  environments.
- **No Nested Parallelism**: The `Experiment` class is the sole manager of the
  `multiprocessing.Pool`. To prevent deadlocks and undefined behavior, any
  fitter called by a worker *must* be run in its sequential mode (e.g.,
  `StandardNLSFitter` with `parallel=False`).
"""

from deepfmkit import core as dfm
from deepfmkit.physics import SimConfig
from deepfmkit.factories import ExperimentFactory

import numpy as np
import itertools
import multiprocessing
import os
import copy
from tqdm import tqdm
from typing import Optional, Callable, Dict, Any, List, Union, Set, Tuple
import pickle
import logging


def _run_single_trial(job_packet: Tuple) -> Dict[str, Any]:
    """Worker function for parallel processing that executes one full trial.

    This function is designed to be the target of a `multiprocessing.Pool` and
    is defined at the top level to ensure it can be pickled. It unpacks a job
    packet, configures the physics using the provided factory, runs a brief
    simulation, performs all requested analyses on the resulting data, and
    returns a dictionary of results.

    Parameters
    ----------
    job_packet : tuple
        A tuple containing all necessary information for a single trial:
        - trial_params (dict): All parameters (static, axis, stochastic) for
          this specific trial.
        - config_factory (ExperimentFactory): The factory instance used to
          generate physics objects from `trial_params`.
        - analyses_to_run (list[dict]): A list of analysis configurations to
          be performed on the simulated data.
        - n_fit_cycles (int): The number of modulation cycles (`n`) to simulate
          and use for the fit in this trial.
        - f_samp (float): The sampling frequency in Hz.
        - trial_num (int): A unique seed for this trial's noise generation,
          ensuring reproducibility.

    Returns
    -------
    dict
        A dictionary packet containing the input parameters (`point_params`)
        and the results from all analyses (`results`) for this single trial.
    """
    # Unpack the job packet
    (
        trial_params,
        config_factory,
        analyses_to_run,
        num_fit_buffers,
        f_samp,
        trial_num,
    ) = job_packet

    # Create the physics configurations for this trial
    configs = config_factory(trial_params)
    laser_config = configs["laser_config"]

    # Calculate R (raw samples per fit buffer) based on the sampling frequency
    R = int(f_samp / laser_config.fm)

    # Calculate the total number of raw samples needed for this trial.
    num_samples_needed = num_fit_buffers * R
    if num_samples_needed == 0:
        # Handle edge case where calculated samples needed is zero
        logging.warning(
            "Calculated num_samples_needed is zero. Setting to 1 to avoid division by zero."
        )
        num_samples_needed = 1

    # Calculate the actual simulation time in seconds.
    n_seconds_to_simulate = num_samples_needed / f_samp

    # Instantiate and run the simulation using a local DFF object
    dff_local = dfm.DeepFrame()
    main_sim_label = "main"
    main_channel_sim = SimConfig(
        label=main_sim_label,
        laser_config=configs["laser_config"],
        ifo_config=configs["main_ifo_config"],
        f_samp=f_samp,
    )
    dff_local.sims[main_sim_label] = main_channel_sim

    witness_sim_label = None
    if "witness_ifo_config" in configs:
        witness_sim_label = "witness"
        witness_channel_sim = SimConfig(
            label=witness_sim_label,
            laser_config=configs["laser_config"],
            ifo_config=configs["witness_ifo_config"],
            f_samp=f_samp,
        )
        dff_local.sims[witness_sim_label] = witness_channel_sim

    dff_local.simulate(
        label=main_sim_label,
        n_seconds=n_seconds_to_simulate,
        mode="asd",
        trial_num=trial_num,
        witness_label=witness_sim_label,
    )

    # Run the requested analyses
    trial_results = {}
    for analysis in analyses_to_run:
        analysis_name = analysis["name"]
        if analysis_name == "groundtruth":
            # This is not a real fit. Just record the true simulation parameters.
            trial_results[analysis_name] = {
                "m": main_channel_sim.m,
                "phi": main_channel_sim.ifo.phi,
                "psi": main_channel_sim.laser.psi,
                "amp": main_channel_sim.laser.amp,
                # Add any other ground truth parameters you want to save here
            }
            # Skip to the next analysis in the list
            continue

        # This is the existing logic for all other real fitter-based analyses
        fitter_args = copy.deepcopy(analysis.get("fitter_kwargs", {}))
        fitter_args.update(
            {
                "method": analysis["fitter_method"],
                "main_label": "main",  # Use the local label
            }
        )
        if analysis["fitter_method"] in ["nls", "ekf"]:
            fitter_args["parallel"] = False

        fitter_args["n"] = num_fit_buffers
        fitter_args["init_a"] = main_channel_sim.laser.amp
        fitter_args["init_phi"] = main_channel_sim.ifo.phi
        fitter_args["init_psi"] = main_channel_sim.laser.psi
        fitter_args["init_m"] = main_channel_sim.m

        fit_obj = dff_local.fit(**fitter_args)

        # Extract and store the results for this analysis
        if fit_obj:
            results_df = dff_local.fits_df[fit_obj.label]
            mean_results = results_df.mean().to_dict()
            trial_results[analysis_name] = mean_results
        else:
            trial_results[analysis_name] = {}

    # Return the results packet
    return {"point_params": trial_params, "results": trial_results}


class Experiment:
    """A declarative framework for defining and running simulation experiments.

    This class provides a high-level API to define multi-dimensional parameter
    sweeps and Monte Carlo trials, run them in parallel, and aggregate the
    results for analysis and plotting.

    Parameters
    ----------
    description : str, optional
        A brief description of the experiment, used for titles and logging.
    filename : str, optional
        If provided, loads results from this file upon initialization using
        `pickle`.
    seed : int, optional
        A seed for NumPy's global random number generator to ensure
        reproducibility of any stochastic variables defined in the experiment.

    Attributes
    ----------
    axes : dict[str, np.ndarray]
        A dictionary mapping parameter names to arrays of values to be swept.
    static_params : dict[str, Any]
        A dictionary of parameters that remain constant across all trials.
    stochastic_vars : dict[str, dict]
        A dictionary defining stochastic variables for Monte Carlo trials.
    config_factory : ExperimentFactory, optional
        The factory object that generates physics configurations for each trial.
    analyses : list[dict]
        A list of analysis configurations to be run on each trial's data.
    n_trials : int
        The number of Monte Carlo trials to run for each point on the grid of axes.
    n_fit_cycles_per_trial : int
        The number of modulation cycles `n` to simulate and use for fitting
        in each individual trial.
    f_samp : float
        The sampling frequency (Hz) for all simulations in the experiment.
    results : dict[str, Any], optional
        A nested dictionary containing the aggregated results after `run()`
        is called. The structure is designed for easy plotting and analysis.
    """

    def __init__(
        self,
        description: str = "Unnamed Experiment",
        filename: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.description = description
        self.seed = seed
        self.axes: Dict[str, np.ndarray] = {}
        self.static_params: Dict[str, Any] = {}
        self.stochastic_vars: Dict[str, Dict[str, Any]] = {}
        self.config_factory: Optional[ExperimentFactory] = None
        self._expected_params_keys: Set[str] = (
            set()
        )  # Store expected parameters for validation
        self.analyses: List[Dict[str, Any]] = []
        self.n_trials: int = 1
        self.n_fit_buffers_per_trial: int = 10
        self.f_samp: int = 200000
        self.results: Optional[Dict[str, Any]] = (
            None  # To store aggregated results after run()
        )

        if filename is not None:
            self.load_results(filename)

    def _validate_param_name(self, name: str):
        """Internal helper to validate if a parameter name is expected by the factory."""
        if not self._expected_params_keys:
            logging.warning(
                "No config factory set yet. Parameter validation will be skipped until set_config_factory() is called."
            )
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
        """Adds a parameter axis to the experiment's sweep space.

        Parameters
        ----------
        name : str
            The name of the parameter to sweep. Must be recognized by the
            configured `ExperimentFactory`.
        values : np.ndarray
            A NumPy array of the values to sweep over for this parameter.
        """
        self._validate_param_name(name)
        self.axes[name] = np.asarray(values)

    def set_static(self, params: Dict[str, Any]):
        """Sets parameters that remain constant across all trials.

        Parameters
        ----------
        params : dict
            A dictionary of parameter names and their constant values.
        """
        for name in params.keys():
            self._validate_param_name(name)
        self.static_params.update(params)

    def add_stochastic_variable(
        self, name: str, generator_func: Callable, depends_on: Optional[str] = None
    ):
        """Adds a stochastic variable for Monte Carlo trials.

        Parameters
        ----------
        name : str
            The name of the stochastic parameter.
        generator_func : Callable
            A function that, when called, returns a single random value for
            the parameter. If `depends_on` is specified, this function must
            accept the dependency's value as an argument.
        depends_on : str, optional
            The name of another parameter (typically an axis) that the
            `generator_func` depends on. Defaults to None.
        """
        self._validate_param_name(name)
        if depends_on is not None:
            self._validate_param_name(depends_on)  # Also validate the dependency
        self.stochastic_vars[name] = {
            "generator": generator_func,
            "depends_on": depends_on,
        }

    def set_config_factory(self, factory: ExperimentFactory):
        """Sets the factory object that generates physics configurations.

        The factory must be an instance of a class that inherits from
        `ExperimentFactory`. This is a critical step to ensure the experiment
        logic is pickleable for parallel processing.

        Parameters
        ----------
        factory : ExperimentFactory
            An instance of a user-defined factory class.
        """
        if not isinstance(factory, ExperimentFactory):
            raise TypeError(
                "factory must be an instance of a class that inherits from ExperimentFactory."
            )
        self.config_factory = factory
        # Once the factory is set, retrieve its expected parameters for validation
        self._expected_params_keys = self.config_factory._get_expected_params_keys()
        logging.info(
            f"Config factory '{type(factory).__name__}' set. Expected parameters: {sorted(list(self._expected_params_keys))}"
        )

    def add_analysis(
        self,
        name: str,
        fitter_method: str,
        result_cols: Optional[List[str]] = None,
        fitter_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Defines an analysis to be run on each simulated dataset.

        Parameters
        ----------
        name : str
            A unique name for this analysis (e.g., 'nls_fit', 'ekf_results').
        fitter_method : str
            The method string corresponding to a fitter class (e.g., 'nls', 'ekf').
        result_cols : list of str, optional
            A list of column names to collect from the fitter's result
            DataFrame. If None, all available columns are collected.
        fitter_kwargs : dict, optional
            A dictionary of additional keyword arguments to pass to the fitter.
        """
        self.analyses.append(
            {
                "name": name,
                "fitter_method": fitter_method,
                "result_cols": result_cols,
                "fitter_kwargs": fitter_kwargs or {},
            }
        )

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
        np.random.seed(0)  # Use a fixed seed for reproducible "random" values.

        for var_name, var_info in self.stochastic_vars.items():
            generator = var_info["generator"]
            dependency_name = var_info.get("depends_on")

            if dependency_name:
                # If the generator depends on an axis value, pass it.
                if dependency_name not in params:
                    # This is a safeguard against configuration errors.
                    np.random.set_state(
                        original_random_state
                    )  # Restore state before raising
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
        filtered_params = {
            k: v
            for k, v in params.items()
            if k in self._expected_params_keys or k.startswith("_exp_")
        }
        return filtered_params

    def save_results(self, filename: str):
        """Save current experiment results to disk."""
        if self.results is None:
            raise RuntimeError("No results to save. Run the experiment first.")
        with open(filename, "wb") as f:
            pickle.dump(self.results, f)
        logging.info(f"Experiment results saved to {filename}")

    def load_results(self, filename: str):
        """Load experiment results from disk into the object."""
        with open(filename, "rb") as f:
            self.results = pickle.load(f)
        logging.info(f"Experiment results loaded from {filename}")

    def run(
        self, n_cores: Optional[int] = None, filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes the defined experiment, parallelizing over all individual trials.

        This method orchestrates the entire experiment by:
        1.  **Job Generation**: Creating a complete list of all unique parameter
            combinations (for all axes points and all Monte Carlo trials). Each
            combination is packaged into a "job packet".
        2.  **Parallel Execution**: Distributing these job packets to a pool of
            worker processes, which execute the `_run_single_trial` function.
        3.  **Result Aggregation**: Collecting the results from all workers and
            restructuring the flat list of results into N-dimensional NumPy
            arrays that match the experiment's axes. Summary statistics (mean,
            std, etc.) are computed across the trial dimension.

        Parameters
        ----------
        n_cores : int, optional
            The number of CPU cores to use for parallel execution. If `None`,
            it defaults to the number of available cores reported by `os.cpu_count()`.
        filename : str, optional
            If provided, saves the final aggregated results dictionary to this
            file using `pickle`.

        Returns
        -------
        dict
            A nested dictionary containing the aggregated results. The structure is:
            `results['axes']`: The axes definitions.
            `results[analysis_name][param_name]['mean'/'std'/'all_trials']`:
            The final arrays, whose shapes match the experiment's axes.

        Raises
        ------
        ValueError
            If a configuration factory has not been set using `set_config_factory`.
        """
        # --- 0. Pre-flight Checks ---
        if self.config_factory is None:
            raise ValueError(
                "A configuration factory must be set using set_config_factory()."
            )
        if not self.axes and not self.n_trials > 0:
            raise ValueError(
                "At least one parameter axis must be defined using add_axis(), or n_trials must be > 0."
            )

        if self.seed is not None:
            np.random.seed(self.seed)
            logging.info(
                f"Seeding NumPy's global random number generator with seed: {self.seed}"
            )

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
                trial_params["_exp_point_idx"] = point_indices_tuple
                trial_params["_exp_trial_idx"] = j

                # Generate and add the stochastic variables for this specific trial
                for var_name, var_info in self.stochastic_vars.items():
                    generator = var_info["generator"]
                    dependency_name = var_info.get("depends_on")
                    if dependency_name:
                        trial_params[var_name] = generator(
                            trial_params[dependency_name]
                        )
                    else:
                        trial_params[var_name] = generator()

                # Filter trial_params to include only those keys that the factory expects
                # along with internal experiment indices. This prevents passing unexpected
                # parameters to the factory, which might cause errors or confusion.
                trial_params_for_factory = {
                    k: v
                    for k, v in trial_params.items()
                    if k in self._expected_params_keys or k.startswith("_exp_")
                }

                # Create the final job "packet" (a simple tuple) and add it to the list.
                # This packet is guaranteed to be pickleable.
                job_packets.append(
                    (
                        trial_params_for_factory,
                        self.config_factory,
                        self.analyses,
                        self.n_fit_buffers_per_trial,
                        self.f_samp,
                        trial_counter,
                    )
                )
                trial_counter += 1

        logging.info(
            f"Starting experiment '{self.description}' with {len(job_packets)} total trials."
        )
        logging.info(f"Using {n_cores} CPU cores for parallel processing.")

        # --- 2. Execute All Jobs in Parallel ---
        # The 'with' statement ensures the pool is properly closed.
        with multiprocessing.Pool(processes=n_cores) as pool:
            # pool.imap is an iterator that yields results as they complete.
            # tqdm wraps this iterator to create a live progress bar.
            flat_results = list(
                tqdm(
                    pool.imap(_run_single_trial, job_packets),
                    total=len(job_packets),
                    desc="Running Trials",
                )
            )

        # --- 3. Aggregate the Results ---
        # Initialize the top-level results dictionary
        results = {"axes": self.axes}

        # Determine the columns to collect for each analysis
        for analysis in self.analyses:
            analysis_name = analysis["name"]
            results[analysis_name] = {}
            cols_to_collect = analysis.get("result_cols")

            if cols_to_collect is None:
                all_keys = set()
                for res_packet in flat_results:
                    if analysis_name in res_packet["results"]:
                        all_keys.update(res_packet["results"][analysis_name].keys())
                cols_to_collect = sorted(list(all_keys))

            shape = (
                tuple(len(ax) for ax in self.axes.values()) + (self.n_trials,)
                if self.axes
                else (self.n_trials,)
            )

            for col in cols_to_collect:
                # Check the type of the first piece of data for this column.
                # If it's not a standard number type, initialize the array with dtype=object.
                first_val = None
                for res_packet in flat_results:
                    if (
                        analysis_name in res_packet["results"]
                        and col in res_packet["results"][analysis_name]
                    ):
                        first_val = res_packet["results"][analysis_name][col]
                        break

                # Default to float, but switch to object if we find a non-numeric type
                grid_dtype = float
                if not isinstance(first_val, (int, float, np.number)):
                    grid_dtype = object

                all_trials_grid = np.full(
                    shape, np.nan if grid_dtype is float else None, dtype=grid_dtype
                )
                results[analysis_name][col] = {"all_trials": all_trials_grid}

        # Populate the final grids directly from the flat list of results
        for packet in flat_results:
            # Retrieve the N-dimensional index and trial index from the packet
            point_idx = packet["point_params"]["_exp_point_idx"]
            trial_idx = packet["point_params"]["_exp_trial_idx"]
            # Combine them to get the full index for the final grid
            if not axis_names:  # Handle case of no axes, just trials
                full_idx = (trial_idx,)
            else:
                full_idx = point_idx + (trial_idx,)

            for analysis in self.analyses:
                analysis_name = analysis["name"]
                if analysis_name in packet["results"]:
                    # Place the result for each parameter into its correct grid location
                    for col, stats_dict in results[analysis_name].items():
                        val = packet["results"][analysis_name].get(col, np.nan)
                        stats_dict["all_trials"][full_idx] = val

        # Calculate summary statistics (mean, std) across the trial dimension
        for analysis_name, res_dict in results.items():
            if analysis_name == "axes":
                continue
            for col, stats_dict in res_dict.items():
                all_trials_data = stats_dict["all_trials"]

                # Only calculate stats for numeric data.
                if all_trials_data.dtype == object:
                    stats_dict["mean"] = None
                    stats_dict["std"] = None
                    stats_dict["min"] = None
                    stats_dict["max"] = None
                    stats_dict["worst"] = None
                    continue  # Skip to the next column

                # This part is for numeric data only
                stats_dict["mean"] = np.nanmean(all_trials_data, axis=-1)
                stats_dict["std"] = np.nanstd(all_trials_data, axis=-1)
                stats_dict["min"] = np.nanmin(stats_dict["all_trials"], axis=-1)
                stats_dict["max"] = np.nanmax(stats_dict["all_trials"], axis=-1)

                # Define "worst-case" as the most extreme deviation from the mean

                # The 'mean' can be NaN if all trials for a point were NaN. Handle this.
                mean_vals = stats_dict["mean"]
                # Create a version of the mean that can be broadcast, replacing NaNs with a neutral value like 0
                mean_broadcastable = np.nan_to_num(mean_vals[..., np.newaxis])

                deviation = np.abs(stats_dict["all_trials"] - mean_broadcastable)

                # Find the indices of the maximum deviation.
                # `nanargmax` will fail if an entire slice is NaN, so we must handle it.
                # We will find where all trials are NaN along the last axis.
                all_nan_mask = np.all(np.isnan(deviation), axis=-1)

                # For slices that are all NaN, nanargmax would raise an error.
                # We can temporarily fill these slices with a value (e.g., 0) so the function runs,
                # then use the mask to put NaN back where it belongs.
                deviation_filled = np.nan_to_num(deviation, nan=0.0)
                worst_indices = np.argmax(deviation_filled, axis=-1)

                # Get the worst-case values using the found indices
                worst_case = np.take_along_axis(
                    stats_dict["all_trials"], worst_indices[..., np.newaxis], axis=-1
                ).squeeze(-1)

                # Where all trials were NaN, the worst case must also be NaN.
                worst_case[all_nan_mask] = np.nan

                stats_dict["worst"] = worst_case

        logging.info("Experiment run complete. Results aggregated.")
        self.results = results  # Store results for potential plotting/inspection

        if filename is not None:
            self.save_results(filename)
            logging.info(f"Results saved to: {filename}")

        return results

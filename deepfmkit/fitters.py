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
"""A collection of parameter estimation algorithms for DFMI data.

This module implements the Strategy design pattern for signal processing. A
`BaseFitter` abstract class defines a common interface, and concrete subclasses
provide specific implementations of different readout algorithms. This allows
the user to select an algorithm at runtime via the `DeepFrame.fit()` method.

Available Fitters:
- StandardNLSFitter: A high-performance Non-Linear Least Squares
  fitter operating in the frequency domain.
- EKFFitter: A time-domain Extended Kalman Filter for real-time state
  tracking.
"""

from deepfmkit.fit import fit, calculate_quadratures, ALL_PARAMS, DEFAULT_GUESS
from deepfmkit.data import RawData

import os
import math
from multiprocessing import Pool
from scipy.optimize import minimize_scalar
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import constants as sc
import logging
from typing import Any, Dict, List, Tuple


def _process_fit_chunk(args: tuple) -> List[Dict[str, Any]]:
    """Worker function for parallel NLS fitting.

    This top-level function is designed to be pickled and sent to worker
    processes. It receives a chunk of raw data buffers and processes them
    sequentially, applying a "warm-start" optimization where the result of
    one buffer serves as the initial guess for the next.

    Parameters
    ----------
    args : tuple
        A tuple containing all necessary arguments:
        - raw_data_chunk (np.ndarray): A 2D array of raw data, shape (n_buffers, R).
        - initial_guess_active (list[float]): The initial parameter guess for the active parameters.
        - R (int): The number of samples per buffer.
        - ndata (int): The number of harmonics to use in the fit.
        - fm (float): The modulation frequency in Hz.
        - f_samp (float): The sampling frequency in Hz.
        - fit_params (list[str]): The list of parameter names being actively fitted.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries, where each dictionary holds the fit result
        for a single buffer in the chunk.
    """
    # 1. Unpack arguments
    raw_data_chunk, initial_guess_active, R, ndata, fm, f_samp, fit_params = args

    # 2. Setup
    results_list = []
    current_guess_active = np.array(initial_guess_active)
    w0 = 2.0 * np.pi * fm / f_samp

    # 3. Process the chunk sequentially
    for i in range(raw_data_chunk.shape[0]):
        buffer_data = raw_data_chunk[i]

        QI_data_mean = np.zeros(2 * ndata)
        for n in range(ndata):
            Q_data, I_data = calculate_quadratures(n, buffer_data, w0)
            QI_data_mean[n] = Q_data.mean()
            QI_data_mean[n + ndata] = I_data.mean()

        # Pass the active guess and param list to the low-level fit function
        status, fit_parm_full, fit_ssq = fit(
            ndata, QI_data_mean, current_guess_active, fit_params
        )

        # Update the guess for the next iteration using the newly fitted active parameters
        current_guess_dict = {name: val for name, val in zip(ALL_PARAMS, fit_parm_full)}
        current_guess_active = [current_guess_dict[p_name] for p_name in fit_params]

        results_list.append(
            {
                "amp": fit_parm_full[0],
                "m": fit_parm_full[1],
                "phi": fit_parm_full[2],
                "psi": fit_parm_full[3],
                "dc": np.mean(buffer_data),
                "ssq": fit_ssq,
                "fitok": status,
            }
        )

    return results_list


def _calculate_fit_params(raw_obj: RawData, n: int) -> Tuple[int, float, int]:
    """Calculates buffer and rate parameters for a fit."""
    # --- Basic validations ---
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"`n` must be a positive integer, got {n!r}.")
    if not (math.isfinite(raw_obj.f_samp) and raw_obj.f_samp > 0):
        raise ValueError(f"`f_samp` must be finite and > 0, got {raw_obj.f_samp!r}.")
    if not (math.isfinite(raw_obj.fm) and raw_obj.fm > 0):
        raise ValueError(f"`fm` must be finite and > 0, got {raw_obj.fm!r}.")
    if raw_obj.data is None or getattr(raw_obj.data, "shape", None) is None:
        raise TypeError("`raw_obj.data` must be an array-like with a .shape attribute.")

    # --- Compute target buffer size (in samples) ---
    R_float = (raw_obj.f_samp / raw_obj.fm) * n
    if not math.isfinite(R_float) or R_float < 1:
        raise ValueError(
            "Requested configuration yields <1 sample per buffer. "
            f"Increase `n` or decrease `fm` (got R≈{R_float:.3g})."
        )
    # Use rounding to avoid edge cases from floating-point noise.
    R = int(round(R_float))
    R = max(R, 1)

    # --- Fit rate and number of full buffers ---
    fs = raw_obj.f_samp / R
    N = int(raw_obj.data.shape[0])
    if N < R:
        raise ValueError(
            "Buffer configuration results in 0 full buffers: "
            f"need at least R={R} samples, but data has N={N}. "
            "Increase data length or reduce `n`."
        )
    nbuf = N // R
    return R, fs, nbuf


class BaseFitter(ABC):
    """Abstract base class for all DFMI fitting algorithms.

    This class defines the common interface for all fitters, ensuring they can
    be used interchangeably by the `DeepFrame` controller (Strategy pattern).

    Attributes
    ----------
    config : dict
        A dictionary of common fitting parameters, must include 'n' (the
        number of modulation cycles per buffer).
    """

    def __init__(self, fit_config: Dict[str, Any]):
        """Initializes the fitter with a common configuration.

        Parameters
        ----------
        fit_config : dict
            A dictionary of fitting parameters.
        """
        self.config = fit_config
        if "n" not in self.config:
            raise ValueError("Fit configuration must include 'n'.")

    @abstractmethod
    def fit(self, main_raw: RawData, **kwargs: Any) -> pd.DataFrame:
        """The main fitting method to be implemented by all subclasses.

        This method performs the core fitting logic on the provided raw data
        and returns the results as a structured pandas DataFrame.

        Parameters
        ----------
        main_raw : RawData
            The raw data object for the primary channel to be fitted.
        **kwargs : Any
            Algorithm-specific keyword arguments.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the time-series of the fitted parameters.
            Must include columns: 'amp', 'm', 'phi', 'psi', 'dc', 'ssq', 'fitok'.
            May also include 'tau' for relevant fitters.
        """
        pass


class EKFFitter(BaseFitter):
    """A fitter that performs state estimation using an Extended Kalman Filter.

    This fitter operates in the time domain, updating its state estimate with
    every incoming data sample. It uses a random walk process model, a common
    and robust choice when the exact parameter dynamics are unknown.
    """

    def fit(self, main_raw: RawData, **kwargs: Any) -> pd.DataFrame:
        """Processes raw data sequentially using an EKF.

        Parameters
        ----------
        main_raw : RawData
            The raw data object for the primary channel.
        **kwargs : Any
            Keyword arguments for EKF initialization and tuning:
            - init_a, init_m, init_phi, init_psi (float): Initial state guesses.
            - P0_diag (list[float]): Diagonal of the initial state covariance `P`.
            - Q_diag (list[float]): Diagonal of the process noise covariance `Q`.
            - R_val (float): Measurement noise variance `R`.
            - verbose (bool): If True, shows a progress bar.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the EKF state estimates over time.
        """
        # --- 0. Unpack Configuration and Data ---
        data = main_raw.data["ch0"].to_numpy()

        # I'll unpack initial conditions from kwargs, with sensible defaults.
        init_a = kwargs.get("init_a", 1.6)
        init_m = kwargs.get("init_m", 6.0)
        init_phi = kwargs.get("init_phi", 0.0)
        init_psi = kwargs.get("init_psi", 0.0)
        P0_diag = kwargs.get("P0_diag", [1.0] * 5)
        Q_diag = kwargs.get("Q_diag", [1e-8, 1e-8, 1e-6, 1e-6, 1e-8])
        R_val = kwargs.get("R_val", None)
        verbose = kwargs.get("verbose", True)

        # --- 1. EKF Initialization ---
        # State vector: x = [amplitude, mod_depth, phase, mod_phase, dc_offset]
        dim_x = 5
        x = np.array([init_a, init_m, init_phi, init_psi, np.mean(data)])
        P = np.diag(P0_diag)
        Q = np.diag(Q_diag)
        if R_val is None:
            R_val = np.var(data)
        R = np.array([[R_val]])
        F = np.eye(dim_x)

        # Setup for the run
        n_samp = len(data)
        w_m = 2 * np.pi * main_raw.fm
        t_axis = np.arange(n_samp) / main_raw.f_samp

        # Calculate downsampling parameters to match NLS fitter output rate
        R_downsample, fs, nbuf = _calculate_fit_params(main_raw, self.config["n"])

        results = np.zeros((nbuf, dim_x))
        logging.debug(f"Running EKF for {n_samp} samples...")

        # --- 2. Main EKF Loop ---
        loop_iter = (
            tqdm(range(n_samp), desc="EKF Progress") if verbose else range(n_samp)
        )

        for k in loop_iter:
            # PREDICT STEP
            P = F @ P @ F.T + Q

            # UPDATE STEP
            a, m, phi, psi, dc = x
            theta = w_m * t_axis[k] + psi
            full_phase_arg = phi + m * np.cos(theta)

            h_x = a * np.cos(full_phase_arg) + dc  # Model prediction

            # Jacobian of measurement model H = [dh/da, dh/dm, dh/dphi, dh/dpsi, dh/ddc]
            sin_full_arg = np.sin(full_phase_arg)
            H = np.array(
                [
                    [
                        np.cos(full_phase_arg),
                        -a * sin_full_arg * np.cos(theta),
                        -a * sin_full_arg,
                        +a * m * sin_full_arg * np.sin(theta),
                        1.0,
                    ]
                ]
            )

            # Innovation (residual) and Kalman Gain
            y_k = data[k] - h_x
            S = H @ P @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)

            # Update state and covariance
            x = x + (K @ y_k.reshape(1, 1)).flatten()
            P = (np.eye(dim_x) - K @ H) @ P

            # Store result at the downsampled rate
            if (k + 1) % R_downsample == 0:
                buf_idx = (k + 1) // R_downsample - 1
                if buf_idx < nbuf:
                    results[buf_idx, :] = x

        # --- 3. Create and return results DataFrame ---
        logging.debug("EKF processing finished. Packaging results...")

        df_dict = {
            "amp": results[:, 0],
            "m": results[:, 1],
            "phi": results[:, 2],
            "psi": results[:, 3],
            "dc": results[:, 4],
            "ssq": np.zeros(nbuf),
            "fitok": np.ones(nbuf, dtype=int),
        }

        return pd.DataFrame(df_dict)


class StandardNLSFitter(BaseFitter):
    """A fitter using the standard frequency-domain Non-Linear Least Squares method.

    This class encapsulates a high-performance, block-parallel fitting
    strategy. It includes robust initialization routines to improve
    convergence and supports "constrained" fits where a subset of parameters
    can be held constant.
    """

    def fit(self, main_raw: RawData, **kwargs: Any) -> pd.DataFrame:
        """Performs the NLS fit on raw data.

        This method orchestrates the fitting process, including optional smart
        initialization, data buffering, and dispatching to a sequential or
        parallel executor.

        Parameters
        ----------
        main_raw : RawData
            The raw data object for the primary channel.
        **kwargs : Any
            - ndata (int): Number of harmonics to fit. Defaults to 10.
            - fit_params (list[str]): List of parameters to fit (e.g.,
              ['amp', 'm', 'phi']). Defaults to all four.
            - init_psi_method (str): Method for smart initialization of psi
              ('scan' or 'minimize'). If None, standard guess is used.
            - parallel (bool): If True, run in parallel. Defaults to True.
            - n_cores (int): Number of CPU cores for parallel execution.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the time-series of fitted parameters.
        """
        # --- 0. Unpack Configuration ---
        n = self.config["n"]
        ndata = kwargs.get("ndata", self.config.get("ndata", 10))
        if "ndata" in kwargs:
            ndata = kwargs["ndata"]
            kwargs.pop("ndata")

        # Determine which parameters to fit. Defaults to all four standard parameters.
        fit_params = kwargs.get("fit_params", ALL_PARAMS)

        # Unpack execution and initial guess parameters from kwargs.
        parallel = kwargs.get("parallel", True)
        init_a = kwargs.get("init_a", DEFAULT_GUESS["amp"])
        init_m = kwargs.get("init_m", DEFAULT_GUESS["m"])
        init_phi = kwargs.get("init_phi", DEFAULT_GUESS["phi"])

        # Get the method for smart initialization. If None, it will be skipped.
        init_psi_method = kwargs.get("init_psi_method", None)

        # Calculate buffer parameters based on the raw data object.
        R, _, nbuf = _calculate_fit_params(main_raw, n)
        if nbuf == 0:
            logging.warning("Number of buffers is zero. No data to fit.")
            return pd.DataFrame()

        # --- 1. Smart Initialization for `psi` ---
        # If a method is specified, run the initialization routine.
        # This is a crucial step to avoid local minima.
        if init_psi_method and "psi" in fit_params:
            # The _psi_init routine requires a full 4-parameter fit to properly
            # evaluate the SSQ landscape for psi.
            init_psi = self._psi_init(
                main_raw, init_psi_method, init_a, init_m, R, ndata
            )
        else:
            # If smart initialization is disabled or if psi is not a fit parameter,
            # use the default value.
            init_psi = kwargs.get("init_psi", DEFAULT_GUESS["psi"])

        # --- 2. Construct Final Initial Guess ---
        # Create a dictionary of the best-known initial values for all parameters.
        full_initial_guess = {
            "amp": init_a,
            "m": init_m,
            "phi": init_phi,  # Phi is typically started at 0
            "psi": init_psi,  # Use the potentially optimized psi value
        }

        # From the full dictionary, create the vector of only the parameters
        # that will be actively fitted. This vector is passed to the LMA.
        initial_guess_active = [full_initial_guess[p_name] for p_name in fit_params]

        # --- 3. Dispatch to Executor ---
        # Based on the 'parallel' flag, run the fit sequentially or in parallel.
        if parallel:
            return self._fit_parallel(
                main_raw, R, nbuf, ndata, initial_guess_active, fit_params, **kwargs
            )
        else:
            return self._fit_sequential(
                main_raw, R, nbuf, ndata, initial_guess_active, fit_params
            )

    def _fit_sequential(
        self,
        main_raw: RawData,
        R: int,
        nbuf: int,
        ndata: int,
        initial_guess_active: List[float],
        fit_params: List[str],
    ) -> pd.DataFrame:
        """Executes the fit sequentially, one buffer at a time."""
        logging.debug(
            f"Processing '{main_raw.label}' sequentially with fit_params={fit_params}..."
        )

        # Start with the optimized initial guess.
        current_guess_active = np.array(initial_guess_active)
        results_list = []

        # Reshape the entire dataset into a 2D array of [nbuf, R] for efficient iteration.
        raw_data_full = main_raw.data.values.reshape(-1, R)

        # Iterate directly over the chunks of data in raw_data_full.
        for b in range(nbuf):
            # Pass the actual data buffer to the fitting helper function.
            raw_buffer = raw_data_full[b]
            result_dict = self._fit_single_buffer(
                raw_buffer,
                main_raw.fm,
                main_raw.f_samp,
                ndata,
                current_guess_active,
                fit_params,
            )
            results_list.append(result_dict)

            # Use the result from this buffer as a "warm start" for the next one.
            full_result_params = {
                name: result_dict[name] for name in ["amp", "m", "phi", "psi"]
            }
            current_guess_active = [full_result_params[p_name] for p_name in fit_params]

        return pd.DataFrame(results_list)

    def _fit_parallel(
        self,
        main_raw: RawData,
        R: int,
        nbuf: int,
        ndata: int,
        initial_guess_active: List[float],
        fit_params: List[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Executes the fit in parallel across multiple cores."""
        n_cores = kwargs.get("n_cores", os.cpu_count())
        n_cores = min(n_cores, nbuf - 1 if nbuf > 1 else 1)
        logging.debug(
            f"Processing '{main_raw.label}' in parallel with fit_params={fit_params}..."
        )

        # "Seeding": Fit the first buffer sequentially to get a robust starting point.
        # First, we must extract the raw data for the first buffer as a NumPy array.
        raw_buffer_first = main_raw.data.iloc[0:R].values.flatten()

        # Now, call the pure helper function with the correct arguments.
        first_fit_result = self._fit_single_buffer(
            raw_buffer_first,
            main_raw.fm,
            main_raw.f_samp,
            ndata,
            initial_guess_active,
            fit_params,
        )

        # Create the warm-start guess for all other parallel workers.
        seed_guess_dict = {
            name: val
            for name, val in zip(
                ALL_PARAMS, [first_fit_result[k] for k in ["amp", "m", "phi", "psi"]]
            )
        }
        seed_guess_active = [seed_guess_dict[p_name] for p_name in fit_params]

        raw_data_full = main_raw.data.values.reshape(-1, R)
        if nbuf <= 1:
            return pd.DataFrame([first_fit_result])

        # Split the remaining data into chunks for the worker processes.
        chunks = np.array_split(raw_data_full[1:], n_cores)
        job_args = [
            (
                chunk,
                seed_guess_active,
                R,
                ndata,
                main_raw.fm,
                main_raw.f_samp,
                fit_params,
            )
            for chunk in chunks
            if chunk.size > 0
        ]

        chunk_results_list = []
        if job_args:
            with Pool(n_cores) as p:
                chunk_results_list = list(
                    tqdm(
                        p.imap(_process_fit_chunk, job_args),
                        total=len(job_args),
                        desc="Parallel Fit",
                    )
                )

        final_results = [first_fit_result]
        for chunk_res in chunk_results_list:
            final_results.extend(chunk_res)
        return pd.DataFrame(final_results)

    def _fit_single_buffer(
        self,
        raw_buffer: np.ndarray,
        fm: float,
        f_samp: float,
        ndata: int,
        initial_guess_active: np.ndarray,
        fit_params: List[str],
    ) -> Dict[str, Any]:
        """Processes a single buffer of raw data to produce one fit result.

        This is a pure helper function that operates on a provided numpy array.
        """
        # Calculate the I/Q components from the provided data buffer.
        w0 = 2.0 * np.pi * fm / f_samp
        QI_data_mean = np.zeros(2 * ndata)
        for n in range(ndata):
            Q_data, I_data = calculate_quadratures(n, raw_buffer, w0)
            QI_data_mean[n] = Q_data.mean()
            QI_data_mean[n + ndata] = I_data.mean()

        # Call the core NLS fitting routine from fit.py.
        status, fit_parm_full, fit_ssq = fit(
            ndata, QI_data_mean, initial_guess_active, fit_params
        )

        # Package the full 4-parameter result into a dictionary.
        return {
            "amp": fit_parm_full[0],
            "m": fit_parm_full[1],
            "phi": fit_parm_full[2],
            "psi": fit_parm_full[3],
            "dc": np.mean(raw_buffer),
            "ssq": fit_ssq,
            "fitok": status,
        }

    def _psi_init(
        self,
        main_raw: RawData,
        method: str,
        init_a: float,
        init_m: float,
        R: int,
        ndata: int,
    ) -> float:
        """Finds a robust initial guess for the `psi` parameter."""
        logging.debug(f"Initializing psi parameter using '{method}' method...")
        try_psi_args = (main_raw, init_a, init_m, R, ndata)
        if method == "scan":
            psi_candidates = np.linspace(0, 2 * np.pi, 20, endpoint=False)
            ssq_values = [self._try_psi(p, *try_psi_args) for p in psi_candidates]
            final_psi = psi_candidates[np.argmin(ssq_values)]
        elif method == "minimize":
            coarse_psis = np.linspace(0, 2 * np.pi, 4, endpoint=False)
            coarse_ssqs = [self._try_psi(p, *try_psi_args) for p in coarse_psis]
            best_initial_psi = coarse_psis[np.argmin(coarse_ssqs)]
            res = minimize_scalar(
                self._try_psi,
                args=try_psi_args,
                bounds=(best_initial_psi - 1.0, best_initial_psi + 1.0),
                method="bounded",
            )
            final_psi = res.x
        else:
            final_psi = 0.0
        logging.debug(f"Selected init_psi = {final_psi:.4f}")
        return final_psi

    def _try_psi(
        self,
        psi: float,
        main_raw: RawData,
        init_a: float,
        init_m: float,
        R: int,
        ndata: int,
    ) -> float:
        """Helper function to test a single psi value and return the resulting SSQ."""
        initial_guess = np.array([init_a, init_m, 0.0, psi])
        fit_params = ["amp", "m", "phi", "psi"]

        # For the psi test, we only need the first buffer of data.
        raw_buffer_first = main_raw.data.iloc[0:R].values.flatten()

        # Call the updated, pure helper function.
        result_dict = self._fit_single_buffer(
            raw_buffer_first,
            main_raw.fm,
            main_raw.f_samp,
            ndata,
            initial_guess,
            fit_params,
        )
        return result_dict["ssq"]
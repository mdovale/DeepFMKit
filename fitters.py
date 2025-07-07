import os
from multiprocessing import Pool
from scipy.optimize import minimize, minimize_scalar
from .fit import fit, calculate_quadratures
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

def _process_fit_chunk(args):
    """
    Worker function for parallel processing. Fits a chunk of raw data buffers.

    I've moved this function here from core.py to keep all fitting logic
    together. It is a top-level function so that it can be pickled by the
    multiprocessing library. It processes its assigned chunk sequentially
    to maintain the warm-start advantage within the chunk.

    Parameters
    ----------
    args : tuple
        A tuple containing all necessary arguments:
        (raw_data_chunk, initial_guess, R, ndata, f_mod, f_samp)

    Returns
    -------
    list_of_dicts : list
        A list of dictionaries, where each dictionary holds the fit result
        for a single buffer in the chunk.
    """
    # 1. Unpack arguments
    raw_data_chunk, initial_guess, R, ndata, f_mod, f_samp = args
    
    # 2. Setup
    results_list = []
    current_guess = np.array(initial_guess)
    w0 = 2. * np.pi * f_mod / f_samp
    
    # 3. Process the chunk sequentially
    for i in range(raw_data_chunk.shape[0]):
        buffer_data = raw_data_chunk[i]
        
        QI_data_mean = np.zeros(2 * ndata)
        for n in range(ndata):
            Q_data, I_data = calculate_quadratures(n, buffer_data, w0)
            QI_data_mean[n] = Q_data.mean()
            QI_data_mean[n + ndata] = I_data.mean()
        
        status, fit_parm, fit_ssq = fit(ndata, QI_data_mean, current_guess)
        
        current_guess = fit_parm
        
        results_list.append({
            'amp': fit_parm[0], 'm': fit_parm[1], 'phi': fit_parm[2], 'psi': fit_parm[3],
            'dc': np.mean(buffer_data), 'ssq': fit_ssq, 'fitok': status
        })
        
    return results_list

def _calculate_fit_params(raw_obj, n):
    """
    Calculates buffer and rate parameters for a fit.
    This is a standalone utility function, which I refactored from the
    DeepFitFramework.fit_init method to make the fitters self-contained.

    Parameters
    ----------
    raw_obj : DeepRawObject
        The raw data object for which to calculate the parameters.
    n : int
        The number of modulation cycles per fit buffer.

    Returns
    -------
    tuple
        A tuple containing (R, fs, nbuf): the buffer size in samples,
        the fit data rate, and the total number of buffers.
    """
    R = int(raw_obj.f_samp / raw_obj.f_mod * n)
    fs = raw_obj.f_samp / R
    nbuf = int(raw_obj.data.shape[0] / R)
    if nbuf == 0:
        logging.error('Check buffer size! nbuf is zero.')
    return R, fs, nbuf

class BaseFitter(ABC):
    """
    Abstract base class for all DFMI fitting algorithms.

    I created this base class to define a common interface for all my fitters,
    ensuring they can be used interchangeably by the main DeepFitFramework controller.
    This "Strategy" design pattern will make the codebase much more modular
    and extensible for future algorithms like HW-DFMI.
    """
    def __init__(self, fit_config: dict):
        """
        Initializes the fitter with common configuration.

        Parameters
        ----------
        fit_config : dict
            A dictionary of common fitting parameters. It must include 'n' (the
            number of modulation cycles per buffer) and may also include
            other parameters like 'ndata' for NLS fitters.
        """
        self.config = fit_config
        if 'n' not in self.config:
            raise ValueError("Fit configuration must include 'n'.")

    @abstractmethod
    def fit(self, main_raw, **kwargs) -> pd.DataFrame:
        """
        The main fitting method to be implemented by all subclasses.

        This method performs the core fitting logic on the provided raw data
        and returns the results as a structured pandas DataFrame.

        Parameters
        ----------
        main_raw : DeepRawObject
            The raw data object for the primary channel to be fitted.
        **kwargs :
            Algorithm-specific keyword arguments. For example, a W-DFMI fitter
            would expect a `witness_raw` object here.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the time-series of the fitted parameters.
            Must include columns: 'amp', 'm', 'phi', 'psi', 'dc', 'ssq', 'fitok'.
        """
        pass

class EKFFitter(BaseFitter):
    """
    A fitter that performs state estimation using an Extended Kalman Filter.
    """
    def fit(self, main_raw, **kwargs) -> pd.DataFrame:
        """
        Processes raw data sequentially using an EKF.

        I based this implementation on the time-domain EKF theory we discussed.
        It uses a random walk process model, which is a common and robust choice
        when the exact parameter dynamics are unknown.

        Parameters
        ----------
        main_raw : DeepRawObject
            The raw data object for the primary channel to be fitted.
        **kwargs :
            Keyword arguments for EKF initialization:
            - init_a, init_m, init_phi, init_psi (float): Initial guesses.
            - P0_diag (list): Initial state covariance diagonal.
            - Q_diag (list): Process noise covariance diagonal.
            - R_val (float): Measurement noise variance.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the EKF state estimates over time.
        """
        # --- 0. Unpack Configuration and Data ---
        data = main_raw.data["ch0"].to_numpy()
        
        # I'll unpack initial conditions from kwargs, with sensible defaults.
        init_a = kwargs.get('init_a', 1.6)
        init_m = kwargs.get('init_m', 6.0)
        init_phi = kwargs.get('init_phi', 0.0)
        init_psi = kwargs.get('init_psi', 0.0)
        P0_diag = kwargs.get('P0_diag', [1.0] * 5)
        Q_diag = kwargs.get('Q_diag', [1e-8, 1e-8, 1e-6, 1e-6, 1e-8])
        R_val = kwargs.get('R_val', None)

        # --- 1. EKF Initialization ---
        # State vector: x = [amplitude, mod_depth, phase, mod_phase, dc_offset]
        dim_x = 5
        x = np.array([init_a, init_m, init_phi, init_psi, np.mean(data)])
        P = np.diag(P0_diag)
        Q = np.diag(Q_diag)
        if R_val is None: R_val = np.var(data)
        R = np.array([[R_val]])
        F = np.eye(dim_x)
        
        # Setup for the run
        n_samp = len(data)
        w_m = 2 * np.pi * main_raw.f_mod
        t_axis = np.arange(n_samp) / main_raw.f_samp
        
        # Calculate downsampling parameters to match NLS fitter output rate
        R_downsample, fs, nbuf = _calculate_fit_params(main_raw, self.config['n'])

        results = np.zeros((nbuf, dim_x))
        logging.info(f"Running EKF for {n_samp} samples...")
        
        # --- 2. Main EKF Loop ---
        for k in tqdm(range(n_samp), desc="EKF Progress"):
            # PREDICT STEP
            P = F @ P @ F.T + Q

            # UPDATE STEP
            a, m, phi, psi, dc = x
            theta = w_m * t_axis[k] + psi
            full_phase_arg = phi + m * np.cos(theta)

            h_x = a * np.cos(full_phase_arg) + dc # Model prediction
            
            # Jacobian of measurement model H = [dh/da, dh/dm, dh/dphi, dh/dpsi, dh/ddc]
            sin_full_arg = np.sin(full_phase_arg)
            H = np.array([[
                np.cos(full_phase_arg),
                -a * sin_full_arg * np.cos(theta),
                -a * sin_full_arg,
                +a * m * sin_full_arg * np.sin(theta),
                1.0
            ]])

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
        logging.info("EKF processing finished. Packaging results...")
        
        df_dict = {
            'amp': results[:, 0], 'm': results[:, 1], 'phi': results[:, 2],
            'psi': results[:, 3], 'dc': results[:, 4],
            'ssq': np.zeros(nbuf),
            'fitok': np.ones(nbuf, dtype=int)
        }
        
        return pd.DataFrame(df_dict)
    
class StandardNLSFitter(BaseFitter):
    """
    A fitter using the standard frequency-domain Non-Linear Least Squares method.

    This class encapsulates the high-performance, block-parallel fitting
    strategy. It seeds the fit with the first buffer, then processes the
    remaining data in parallel across multiple cores for maximum throughput.
    """
    def fit(self, main_raw, **kwargs) -> pd.DataFrame:
        """
        Performs fit in parallel using the block-wise strategy.

        Parameters
        ----------
        main_raw : DeepRawObject
            The raw data object for the primary channel to be fitted.
        **kwargs :
            Keyword arguments for NLS fitting:
            - init_a, init_m, init_psi (float): Initial guesses.
            - ndata (int): Number of harmonics to fit.
            - n_cores (int): Number of CPU cores to use.
            - init_psi_method (str): 'scan' or 'minimize'.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the NLS fit results over time.
        """
        # --- 0. Unpack Configuration ---
        n = self.config['n']
        ndata = self.config.get('ndata', 10)
        
        init_a = kwargs.get('init_a', 1.6)
        init_m = kwargs.get('init_m', 6.0)
        init_psi = kwargs.get('init_psi', 0.0)
        init_psi_method = kwargs.get('init_psi_method', None)
        n_cores = kwargs.get('n_cores', None)
        
        R, fs, nbuf = _calculate_fit_params(main_raw, n)
        if nbuf == 0: return pd.DataFrame()

        if n_cores is None: n_cores = os.cpu_count()
        n_cores = min(n_cores, nbuf)

        logging.info(f"Processing '{main_raw.label}' with StandardNLSFitter using {n_cores} cores...")

        # --- 1. Seeding: Get a single good initial guess ---
        if init_psi_method:
            psi = self._psi_init(main_raw, init_psi_method, init_a, init_m, R, ndata)
        else:
            psi = init_psi

        seed_guess = np.array([init_a, init_m, 0.0, psi])
        first_fit_result = self._fit_single_buffer(main_raw, 0, R, ndata, seed_guess)
        
        # Use the result of the first fit as the warm start for all parallel workers
        seed_guess = np.array([
            first_fit_result['amp'], first_fit_result['m'], 
            first_fit_result['phi'], first_fit_result['psi']
        ])
        
        # --- 2. Chunking: Split the raw data for each core ---
        raw_data_full = main_raw.data.values.reshape(-1, R)
        if nbuf == 1:
             chunks = [] # No more chunks to process
        else:
             chunks = np.array_split(raw_data_full[1:], n_cores) 
        
        # --- 3. Prepare Jobs for Multiprocessing ---
        job_args = [(chunk, seed_guess, R, ndata, main_raw.f_mod, main_raw.f_samp) 
                    for chunk in chunks if chunk.size > 0]
        
        # --- 4. Run the Pool ---
        chunk_results_list = []
        if job_args:
            with Pool(n_cores) as p:
                chunk_results_list = list(tqdm(p.imap(_process_fit_chunk, job_args), total=len(job_args), desc="Parallel Fit"))
        
        # --- 5. Stitch Results ---
        final_results = [first_fit_result]
        for chunk_res in chunk_results_list:
            final_results.extend(chunk_res)
            
        return pd.DataFrame(final_results)

    def _fit_single_buffer(self, main_raw, b, R, ndata, initial_guess):
        """Processes a single buffer of raw data to produce one fit result."""
        buf = range(b * R, (b + 1) * R)
        raw_buffer = np.array(main_raw.data.iloc[buf]).flatten()
        w0 = 2. * np.pi * main_raw.f_mod / main_raw.f_samp
        
        QI_data_mean = np.zeros(2 * ndata)
        for n in range(ndata):
            Q_data, I_data = calculate_quadratures(n, raw_buffer, w0)
            QI_data_mean[n] = Q_data.mean()
            QI_data_mean[n + ndata] = I_data.mean()
        
        status, fit_parm, fit_ssq = fit(ndata, QI_data_mean, initial_guess)
        
        return {
            'amp': fit_parm[0], 'm': fit_parm[1], 'phi': fit_parm[2], 'psi': fit_parm[3],
            'dc': np.mean(raw_buffer), 'ssq': fit_ssq, 'fitok': status
        }

    def _psi_init(self, main_raw, method, init_a, init_m, R, ndata):
        """Finds an optimal initial guess for the modulation phase (psi)."""
        logging.info(f"Initializing psi parameter using '{method}' method...")

        try_psi_args = (main_raw, init_a, init_m, R, ndata)

        if method == 'scan':
            psi_candidates = np.linspace(0, 2 * np.pi, 20, endpoint=False)
            ssq_values = [self._try_psi(p, *try_psi_args) for p in psi_candidates]
            final_psi = psi_candidates[np.argmin(ssq_values)]

        elif method == 'minimize':
            coarse_psis = np.linspace(0, 2 * np.pi, 4, endpoint=False)
            coarse_ssqs = [self._try_psi(p, *try_psi_args) for p in coarse_psis]
            best_initial_psi = coarse_psis[np.argmin(coarse_ssqs)]
            
            res = minimize_scalar(self._try_psi, args=try_psi_args, bracket=(0.0, 2 * np.pi), 
                                  bounds=(best_initial_psi - 0.5, best_initial_psi + 0.5))
            final_psi = res.x

        else:
            final_psi = 0.0
            
        logging.info(f"Selected init_psi = {final_psi:.4f}")
        return final_psi

    def _try_psi(self, psi, main_raw, init_a, init_m, R, ndata):
        """Helper function to test a single psi value and return the resulting SSQ."""
        initial_guess = np.array([init_a, init_m, 0.0, psi])
        result_dict = self._fit_single_buffer(main_raw, 0, R, ndata, initial_guess)
        return result_dict['ssq']
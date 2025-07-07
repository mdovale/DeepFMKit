from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

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
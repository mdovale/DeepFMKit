import os
from multiprocessing import Pool
from scipy.optimize import minimize, minimize_scalar, least_squares
from scipy.integrate import cumulative_trapezoid
from .fit import fit, calculate_quadratures
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from scipy import constants as sc

def _process_fit_chunk(args):
    """
    Worker function for parallel processing. Fits a chunk of raw data buffers.

    It is a top-level function so that it can be pickled by the
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

def _get_phase_modulation_basis(witness_raw, R, f_samp):
    """
    Processes a witness signal to extract the phase modulation basis waveform.

    This is a helper function that I created to avoid duplicating code across
    all my W-DFMI fitters. It takes the witness signal, removes the DC offset,
    normalizes it, and integrates it to get the phase waveform.

    Parameters
    ----------
    witness_raw : DeepRawObject
        The raw data object for the witness channel.
    R : int
        The buffer size in samples.
    f_samp : float
        The sampling frequency.

    Returns
    -------
    tuple
        A tuple containing (phi_mod_basis, time_axis_buffer).
    """
    # I'll use the first buffer of the witness signal as the definitive template
    witness_buffer_raw = np.array(witness_raw.data.iloc[0:R]).flatten()
    v_w_ac = witness_buffer_raw - np.mean(witness_buffer_raw)
    
    # Witness voltage is proportional to f_mod(t). I correct for the sign inversion
    # at the mid-fringe point and normalize the waveform.
    f_mod_basis = -v_w_ac / np.max(np.abs(v_w_ac))
    
    time_axis_buffer = np.arange(R) / f_samp
    dt = time_axis_buffer[1] - time_axis_buffer[0]
    phi_mod_basis = np.cumsum(f_mod_basis) * dt
    
    return phi_mod_basis, time_axis_buffer

def _get_total_laser_phase(witness_raw, R, f_samp, f_ref):
    """
    Processes a heterodyne witness signal to extract the total laser phase.

    This helper function is specific to the HW-DFMI algorithm. It takes the
    witness signal, which represents the beatnote frequency f_main(t) - f_ref,
    and integrates it to get the total accumulated phase of the main laser.

    Parameters
    ----------
    witness_raw : DeepRawObject
        The raw data object for the HW-DFMI witness channel. Its data is
        assumed to be the instantaneous beatnote frequency in Hz.
    R : int
        The buffer size in samples.
    f_samp : float
        The sampling frequency.
    f_ref : float
        The frequency of the stable reference laser in Hz.

    Returns
    -------
    tuple
        A tuple containing (phi_main, time_axis_buffer).
    """
    # I'll use the first buffer of the witness signal as the definitive template
    # The data is assumed to be the frequency of the beatnote f_beat(t).
    f_beat = np.array(witness_raw.data.iloc[0:R]).flatten()
    
    time_axis_buffer = np.arange(R) / f_samp
    dt = time_axis_buffer[1] - time_axis_buffer[0]
    
    # Reconstruct the main laser's total phase by integrating its frequency.
    # f_main(t) = f_beat(t) + f_ref.
    # phi_main(t) = 2*pi * integral(f_main(t')) dt'.
    # I'll use cumulative trapezoidal integration for better accuracy.
    phi_main = 2 * np.pi * cumulative_trapezoid(f_beat + f_ref, dx=dt, initial=0)
    
    return phi_main, time_axis_buffer

class BaseFitter(ABC):
    """
    Abstract base class for all DFMI fitting algorithms.

    Base class to define a common interface for all fitters,
    ensuring they can be used interchangeably by the main DeepFitFramework controller.
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
        logging.debug(f"Running EKF for {n_samp} samples...")
        
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
        logging.debug("EKF processing finished. Packaging results...")
        
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

        logging.debug(f"Processing '{main_raw.label}' with StandardNLSFitter using {n_cores} cores...")

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
        logging.debug(f"Initializing psi parameter using '{method}' method...")

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
            
        logging.debug(f"Selected init_psi = {final_psi:.4f}")
        return final_psi

    def _try_psi(self, psi, main_raw, init_a, init_m, R, ndata):
        """Helper function to test a single psi value and return the resulting SSQ."""
        initial_guess = np.array([init_a, init_m, 0.0, psi])
        result_dict = self._fit_single_buffer(main_raw, 0, R, ndata, initial_guess)
        return result_dict['ssq']
    
class WDFMI_NLSFitter(BaseFitter):
    """
    * EXPERIMENTAL *
    
    A W-DFMI fitter using a direct Non-Linear Least Squares approach.
    It fits for all 4 parameters (C, tau, phi, psi) simultaneously.
    """
    def fit(self, main_raw, witness_raw, **kwargs) -> pd.DataFrame:
        """
        Performs a fit using the physically exact W-DFMI model.
        """
        # --- 1. Unpack Configuration ---
        n = self.config['n']
        ndata = self.config.get('ndata', 10)
        
        init_a = kwargs.get('init_a', 1.6)
        init_m = kwargs.get('init_m', 6.0)
        init_phi = kwargs.get('init_phi', 0.0)
        init_psi = kwargs.get('init_psi', 0.0)
        
        R, _, nbuf = _calculate_fit_params(main_raw, n)
        if nbuf == 0: return pd.DataFrame()

        laser_cfg = main_raw.sim.laser
        main_ifo_cfg = main_raw.sim.ifo
        omega_mod = 2 * np.pi * laser_cfg.f_mod
        w0_samp = omega_mod / main_raw.f_samp
        
        # --- 2. Process Witness Signal ---
        phi_mod_basis, time_axis_buffer = _get_phase_modulation_basis(witness_raw, R, main_raw.f_samp)
        
        # --- 3. Main Fitting Loop ---
        delta_l = main_ifo_cfg.meas_arml - main_ifo_cfg.ref_arml
        tau_init = delta_l / sc.c
        current_guess = np.array([init_a, tau_init, init_phi, init_psi])
        
        results_list = []
        logging.debug(f"Processing '{main_raw.label}' with WDFMI_NLSFitter...")
        
        for b in range(nbuf):
            buf_range = range(b * R, (b + 1) * R)
            main_buffer_raw = np.array(main_raw.data.iloc[buf_range]).flatten()
            
            # This is the cost function, defined inside the loop to capture the
            # loop-specific `main_buffer_raw` data.
            def _wdfmi_residuals(params, phi_mod_basis_arg, QI_meas):
                C, tau, phi, psi = params
                
                phi_mod_unscaled = 2 * np.pi * laser_cfg.df * phi_mod_basis_arg
                
                t_interp_psi = time_axis_buffer - (-psi / omega_mod)
                phi_mod_shifted = np.interp(t_interp_psi, time_axis_buffer, phi_mod_unscaled, period=time_axis_buffer[-1])
                
                t_interp_tau = time_axis_buffer - tau
                phi_mod_delayed = np.interp(t_interp_tau, time_axis_buffer, phi_mod_shifted, period=time_axis_buffer[-1])
                
                delta_phi_mod = phi_mod_delayed - phi_mod_shifted
                v_model = C * np.cos(phi + delta_phi_mod)
                
                QI_model = np.zeros(2 * ndata)
                for i in range(ndata):
                    n_harm = i + 1
                    QI_model[i] = (2/R) * np.sum(v_model * np.cos(n_harm * w0_samp * np.arange(R)))
                    QI_model[i + ndata] = (2/R) * np.sum(v_model * np.sin(n_harm * w0_samp * np.arange(R)))
                
                return QI_model - QI_meas

            QI_data_meas = np.zeros(2 * ndata)
            for i in range(ndata):
                n_harm = i + 1
                QI_data_meas[i] = (2/R) * np.sum(main_buffer_raw * np.cos(n_harm * w0_samp * np.arange(R)))
                QI_data_meas[i + ndata] = (2/R) * np.sum(main_buffer_raw * np.sin(n_harm * w0_samp * np.arange(R)))

            opt_result = least_squares(
                _wdfmi_residuals, current_guess,
                args=(phi_mod_basis, QI_data_meas), method='lm'
            )
            
            fit_parm = opt_result.x
            C_fit, tau_fit, phi_fit, psi_fit = fit_parm
            m_fit = 2 * np.pi * laser_cfg.df * tau_fit
            current_guess = fit_parm
            
            results_list.append({
                'amp': C_fit, 'm': m_fit, 'phi': phi_fit, 'psi': psi_fit, 
                'dc': np.mean(main_buffer_raw),
                'ssq': np.sum(opt_result.fun**2), 'fitok': 1 if opt_result.success else 0
            })
            
        return pd.DataFrame(results_list)

class WDFMI_OrthogonalFitter(BaseFitter):
    """
    * EXPERIMENTAL *

    Fits W-DFMI data using the robust Orthogonal Demodulation (VarPro) algorithm.
    This is my most robust and definitive W-DFMI implementation.
    """
    def fit(self, main_raw, witness_raw, **kwargs) -> pd.DataFrame:
        """
        Performs a fit using the two-stage Orthogonal Demodulation method.
        """
        # --- 1. Unpack Configuration ---
        n = self.config['n']
        init_m = kwargs.get('init_m', 6.0)
        init_psi = kwargs.get('init_psi', 0.0)
        
        R, _, nbuf = _calculate_fit_params(main_raw, n)
        if nbuf == 0: return pd.DataFrame()

        laser_cfg = main_raw.sim.laser
        main_ifo_cfg = main_raw.sim.ifo
        omega_mod = 2 * np.pi * laser_cfg.f_mod

        # --- 2. Process Witness Signal ---
        phi_mod_basis, time_axis_buffer = _get_phase_modulation_basis(witness_raw, R, main_raw.f_samp)
        phi_mod_unscaled = 2 * np.pi * laser_cfg.df * phi_mod_basis

        # --- 3. Main Fitting Loop ---
        delta_l = main_ifo_cfg.meas_arml - main_ifo_cfg.ref_arml
        tau_init = delta_l / sc.c if laser_cfg.df > 0 else 0.0
        current_guess = np.array([tau_init, init_psi])

        results_list = []
        logging.debug(f"Processing '{main_raw.label}' with WDFMI_OrthogonalFitter...")

        for b in range(nbuf):
            buf_range = range(b * R, (b + 1) * R)
            main_buffer_raw = np.array(main_raw.data.iloc[buf_range]).flatten()
            v_main_ac = main_buffer_raw - np.mean(main_buffer_raw)

            def _get_bases(tau, psi):
                t_interp_psi = time_axis_buffer - (-psi / omega_mod)
                phi_mod_shifted = np.interp(t_interp_psi, time_axis_buffer, phi_mod_unscaled, period=time_axis_buffer[-1])
                t_interp_tau = time_axis_buffer - tau
                phi_mod_delayed = np.interp(t_interp_tau, time_axis_buffer, phi_mod_shifted, period=time_axis_buffer[-1])
                delta_phi_mod = phi_mod_delayed - phi_mod_shifted
                return np.cos(delta_phi_mod), np.sin(delta_phi_mod)

            def _outer_loop_cost(params):
                tau, psi = params
                basis_I, basis_Q = _get_bases(tau, psi)
                A_matrix = np.vstack([basis_I, basis_Q]).T
                _, res, _, _ = np.linalg.lstsq(A_matrix, v_main_ac, rcond=None)
                return res[0]

            opt_result = minimize(_outer_loop_cost, current_guess, method='Nelder-Mead')
            
            tau_fit, psi_fit = opt_result.x
            
            basis_I, basis_Q = _get_bases(tau_fit, psi_fit)
            A_matrix = np.vstack([basis_I, basis_Q]).T
            p_final, final_res, _, _ = np.linalg.lstsq(A_matrix, v_main_ac, rcond=None)
            I_fit, Q_fit = p_final
            
            C_fit = np.sqrt(I_fit**2 + Q_fit**2)
            phi_fit = np.arctan2(-Q_fit, I_fit)
            m_fit = 2 * np.pi * laser_cfg.df * tau_fit
            current_guess = np.array([tau_fit, psi_fit])

            results_list.append({
                'amp': C_fit, 'm': m_fit, 'phi': phi_fit, 'psi': psi_fit,
                'dc': np.mean(main_buffer_raw),
                'ssq': final_res[0] if final_res.size > 0 else 0,
                'fitok': 1 if opt_result.success else 0
            })
            
        return pd.DataFrame(results_list)
    
class WDFMI_SequentialFitter(BaseFitter):
    """
    * EXPERIMENTAL *

    Fits W-DFMI data using the sequential bootstrap algorithm.

    It works by sequentially solving for parameters using the most robust
    method available at each stage.
    """
    def fit(self, main_raw, witness_raw, **kwargs) -> pd.DataFrame:
        """
        Performs a fit using the three-stage sequential bootstrap method.

        1. Finds `tau` via a 1D Variable Projection (VarPro) fit.
        2. With `tau` fixed, finds `psi` via a robust 1D "Differential Phase" fit.
        3. With both `tau` and `psi` known, performs a final linear fit for `C` and `Phi`.

        Parameters
        ----------
        main_raw : DeepRawObject
            The raw data object for the primary channel to be fitted.
        witness_raw : DeepRawObject
            The raw data object for the witness channel.
        **kwargs :
            Keyword arguments for the fit:
            - init_psi (float): Initial guess for the modulation phase.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the fit results over time.
        """
        # --- 1. Unpack Configuration ---
        n = self.config['n']
        init_psi = kwargs.get('init_psi', 0.0)
        
        R, _, nbuf = _calculate_fit_params(main_raw, n)
        if nbuf == 0: return pd.DataFrame()

        laser_cfg = main_raw.sim.laser
        main_ifo_cfg = main_raw.sim.ifo
        omega_mod = 2 * np.pi * laser_cfg.f_mod

        # --- 2. Process Witness Signal ---
        phi_mod_basis, time_axis_buffer = _get_phase_modulation_basis(witness_raw, R, main_raw.f_samp)
        phi_mod_unscaled = 2 * np.pi * laser_cfg.df * phi_mod_basis

        # --- 3. Main Fitting Loop ---
        results_list = []
        logging.debug(f"Processing '{main_raw.label}' with WDFMI_SequentialFitter...")
        
        for b in range(nbuf):
            buf_range = range(b * R, (b + 1) * R)
            main_buffer_raw = np.array(main_raw.data.iloc[buf_range]).flatten()
            v_main_ac = main_buffer_raw - np.mean(main_buffer_raw)
            
            # --- Helper function to construct time-domain basis functions ---
            def get_bases(tau, psi):
                t_interp_psi = time_axis_buffer - (-psi / omega_mod)
                phi_mod_shifted = np.interp(t_interp_psi, time_axis_buffer, phi_mod_unscaled, period=time_axis_buffer[-1])
                t_interp_tau = time_axis_buffer - tau
                phi_mod_delayed = np.interp(t_interp_tau, time_axis_buffer, phi_mod_shifted, period=time_axis_buffer[-1])
                delta_phi_mod = phi_mod_delayed - phi_mod_shifted
                return np.cos(delta_phi_mod), np.sin(delta_phi_mod)

            # --- Stage 1: Find Tau using VarPro ---
            def cost_tau(tau):
                basis_I, basis_Q = get_bases(tau, init_psi) # Use initial psi guess for this stage
                A_matrix = np.vstack([basis_I, basis_Q]).T
                _, res, _, _ = np.linalg.lstsq(A_matrix, v_main_ac, rcond=None)
                return res[0] if res.size > 0 else np.inf
            
            delta_l = main_ifo_cfg.meas_arml - main_ifo_cfg.ref_arml
            tau_init = delta_l / sc.c if laser_cfg.df > 0 else 0.0
            tau_bracket = (tau_init * 0.9, tau_init * 1.1) if tau_init > 0 else (-1e-9, 1e-9)
            res_tau = minimize_scalar(cost_tau, bracket=tau_bracket, method='brent')
            tau_fit = res_tau.x
            
            # --- Stage 2: Find Psi using Differential Phase ---
            ndata_psi = 40 # Number of harmonics for this stage
            w0_samp = omega_mod / main_raw.f_samp
            Q_meas, I_meas = np.zeros(ndata_psi), np.zeros(ndata_psi)
            for i in range(ndata_psi):
                n_harm = i + 1
                Q_meas[i] = (2/R) * np.sum(v_main_ac * np.cos(n_harm * w0_samp * np.arange(R)))
                I_meas[i] = (2/R) * np.sum(v_main_ac * np.sin(n_harm * w0_samp * np.arange(R)))
            alpha_meas = Q_meas + 1j * I_meas

            def get_model_spectrum(psi_trial):
                basis_I, basis_Q = get_bases(tau_fit, psi_trial)
                A_matrix = np.vstack([basis_I, basis_Q]).T
                p_final, _, _, _ = np.linalg.lstsq(A_matrix, v_main_ac, rcond=None)
                I_fit, Q_fit = p_final
                v_model_approx = I_fit * basis_I - Q_fit * basis_Q
                Q_m, I_m = np.zeros(ndata_psi), np.zeros(ndata_psi)
                for i in range(ndata_psi):
                    n_harm = i + 1
                    Q_m[i] = (2/R) * np.sum(v_model_approx * np.cos(n_harm * w0_samp * np.arange(R)))
                    I_m[i] = (2/R) * np.sum(v_model_approx * np.sin(n_harm * w0_samp * np.arange(R)))
                return Q_m + 1j * I_m
                
            def cost_psi(psi_deviation):
                alpha_model = get_model_spectrum(init_psi + psi_deviation)
                phase_error = np.angle(alpha_meas * np.conj(alpha_model))
                return np.var(np.unwrap(phase_error))
                
            res_psi = minimize_scalar(cost_psi, bounds=(-np.pi/2, np.pi/2), method='bounded')
            psi_fit = init_psi + res_psi.x
            
            # --- Stage 3: Final Linear Fit ---
            basis_I, basis_Q = get_bases(tau_fit, psi_fit)
            A_matrix = np.vstack([basis_I, basis_Q]).T
            p_final, final_res, _, _ = np.linalg.lstsq(A_matrix, v_main_ac, rcond=None)
            I_fit, Q_fit = p_final
            
            C_fit = np.sqrt(I_fit**2 + Q_fit**2)
            phi_fit = np.arctan2(-Q_fit, I_fit)
            m_fit = 2 * np.pi * laser_cfg.df * tau_fit
            
            results_list.append({
                'amp': C_fit, 'm': m_fit, 'phi': phi_fit, 'psi': psi_fit,
                'dc': np.mean(main_buffer_raw),
                'ssq': final_res[0] if final_res.size > 0 else 0.0,
                'fitok': 1
            })

        return pd.DataFrame(results_list)
    
class HWDFMI_Fitter(BaseFitter):
    """
    * EXPERIMENTAL *

    Fits HW-DFMI data using the definitive 1D Orthogonal Demodulation algorithm.

    This is the most advanced fitter, designed for the HW-DFMI architecture.
    It leverages a heterodyne witness to measure the laser's total instantaneous
    frequency, which eliminates nearly all free parameters from the model. The
    readout reduces to a highly robust 1D search for the physical time delay `tau`,
    providing a direct measurement of the absolute path length difference.
    """
    def fit(self, main_raw, witness_raw, **kwargs) -> pd.DataFrame:
        """
        Performs a fit using the 1D VarPro method on HW-DFMI data.

        Parameters
        ----------
        main_raw : DeepRawObject
            The raw data object for the primary measurement channel.
        witness_raw : DeepRawObject
            The raw data object for the heterodyne witness. Its data must be
            the instantaneous beatnote frequency, f_main(t) - f_ref.
        **kwargs :
            - init_tau (float, optional): An initial guess for the time delay tau.
              If not provided, it's estimated from the interferometer config.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the fit results.
        """
        # --- 1. Unpack Configuration ---
        n = self.config['n']
        init_tau_kwarg = kwargs.get('init_tau', None)
        
        R, _, nbuf = _calculate_fit_params(main_raw, n)
        if nbuf == 0: return pd.DataFrame()

        # HW-DFMI requires the reference laser frequency, which I'll assume is
        # stored in the witness interferometer config as its 'arml_mod_f'
        # for lack of a better place. This is a convention I'm establishing.
        if hasattr(witness_raw.sim.ifo, 'arml_mod_f'):
            f_ref = witness_raw.sim.ifo.arml_mod_f
        else:
            logging.warning("Reference frequency `f_ref` not found in witness config. Assuming 0 Hz.")
            f_ref = 0.0

        # --- 2. Process Witness Signal ---
        phi_main_template, time_axis_buffer = _get_total_laser_phase(witness_raw, R, main_raw.f_samp, f_ref)

        # --- 3. Main Fitting Loop ---
        if init_tau_kwarg is not None:
            current_tau_guess = init_tau_kwarg
        else:
            delta_l_init = main_raw.sim.ifo.meas_arml - main_raw.sim.ifo.ref_arml
            current_tau_guess = delta_l_init / sc.c

        results_list = []
        logging.debug(f"Processing '{main_raw.label}' with HWDFMI_Fitter...")

        for b in tqdm(range(nbuf), desc="HW-DFMI Fit Progress"):
            buf_range = range(b * R, (b + 1) * R)
            main_buffer_raw = np.array(main_raw.data.iloc[buf_range]).flatten()
            v_main_ac = main_buffer_raw - np.mean(main_buffer_raw)

            # --- Inner function to construct orthogonal bases for a given tau ---
            def get_bases(tau):
                t_interp_delayed = time_axis_buffer - tau
                phi_main_delayed = np.interp(t_interp_delayed, time_axis_buffer, phi_main_template)
                delta_phi_mod = phi_main_template - phi_main_delayed
                return np.cos(delta_phi_mod), np.sin(delta_phi_mod)

            # --- VarPro cost function for the 1D outer loop ---
            def outer_loop_cost(tau):
                basis_I, basis_Q = get_bases(tau)
                A_matrix = np.vstack([basis_I, basis_Q]).T
                # lstsq returns the sum of squared residuals if `b` is a vector
                _, res, _, _ = np.linalg.lstsq(A_matrix, v_main_ac, rcond=None)
                # The residual is an array of size 1, so I extract the scalar
                return res[0] if res.size > 0 else np.inf

            # --- Perform the 1D optimization for tau ---
            tau_bracket = (current_tau_guess * 0.8, current_tau_guess * 1.2) if current_tau_guess != 0 else (-1e-9, 1e-9)
            res_tau = minimize_scalar(outer_loop_cost, bracket=tau_bracket, method='brent')
            tau_fit = res_tau.x
            
            # --- Perform the final linear fit with the optimal tau ---
            basis_I_final, basis_Q_final = get_bases(tau_fit)
            A_matrix_final = np.vstack([basis_I_final, basis_Q_final]).T
            p_final, final_res, _, _ = np.linalg.lstsq(A_matrix_final, v_main_ac, rcond=None)
            Ip_fit, Qp_fit = p_final
            
            # --- Recover physical parameters ---
            C_fit = np.sqrt(Ip_fit**2 + Qp_fit**2)
            phi_fit = np.arctan2(-Qp_fit, Ip_fit)
            
            # The concept of 'm' is less central here, but I'll calculate it
            # for consistency with the DeepFitObject data structure.
            m_effective = np.ptp(phi_main_template - np.interp(time_axis_buffer - tau_fit, time_axis_buffer, phi_main_template)) / 2.0
            
            # There is no 'psi' parameter in this model. It's absorbed into the
            # witness measurement. I will set it to zero.
            psi_fit = 0.0
            
            # Update guess for next buffer for warm start
            current_tau_guess = tau_fit

            results_list.append({
                'amp': C_fit, 'm': m_effective, 'phi': phi_fit, 'psi': psi_fit,
                'dc': np.mean(main_buffer_raw),
                'ssq': final_res[0] if final_res.size > 0 else 0.0,
                'fitok': 1, # minimize_scalar doesn't have a success flag like minimize
                'tau': tau_fit # I'll also store the direct result for tau
            })
            
        return pd.DataFrame(results_list)
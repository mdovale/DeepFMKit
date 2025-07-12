from .fit import coeffs
from .physics import LaserConfig, InterferometerConfig

import numpy as np
from scipy.special import jv
from scipy.special import jv
from scipy.linalg import inv
from scipy.constants import pi, c 

def set_laser_df_for_effect(laser: LaserConfig, ifo: InterferometerConfig, m):
    ref_arml = ifo.ref_arml
    meas_arml = ifo.meas_arml 
    opd = np.abs(meas_arml-ref_arml)
    laser.df = (m * c) / (2 * pi * opd)

def calculate_crlb_for_m(m_true, ndata, snr_db, buffer_size):
    """
    Calculates the Cramér-Rao Lower Bound for the precision of 'm'.
    This calculation is based on the Fisher information matrix and the
    expected noise in the I/Q demodulated data.
    """
    # Signal power of cos(phi) is 0.5
    signal_power = 0.5
    snr_linear = 10**(snr_db / 10.0)
    noise_power_td = signal_power / snr_linear
    
    # Variance of the mean of I/Q data over the buffer
    sigma_iq_sq = noise_power_td / (2 * buffer_size)
    
    # The Fisher matrix is proportional to J^T*J.
    # The absolute amplitude 'a' doesn't matter for the relative precision.
    perfect_params = np.array([1.0, m_true, 0.0, 0.0])
    dummy_data = np.zeros(2 * ndata)
    
    _, JTJ_flat, _ = coeffs(ndata, dummy_data, perfect_params)
    fisher_matrix = JTJ_flat.reshape(4, 4)
    
    try:
        covariance_matrix = np.linalg.inv(fisher_matrix)
        # The variance of 'm' is the (1,1) element of the inverse Fisher matrix,
        # scaled by the noise variance of the I/Q data.
        variance_m = covariance_matrix[1, 1] * sigma_iq_sq
        return np.sqrt(variance_m)
    except np.linalg.LinAlgError:
        return np.nan

def snr_to_asd(snr_db, f_samp):
    """
    Converts a target SNR in dB to an equivalent amplitude noise ASD.
    This assumes the signal is a unit-amplitude cosine with power 0.5.
    """
    signal_power = 0.5 # For a unit amplitude cosine
    snr_linear = 10**(snr_db / 10.0)
    noise_power = signal_power / snr_linear
    # The total noise power is sigma^2. For white noise, Power = ASD^2 * (f_samp / 2).
    # Therefore, ASD = sqrt(Power / (f_samp / 2)).
    asd = np.sqrt(noise_power / (f_samp / 2.0))
    return asd

def calculate_jacobian(ndata, param):
    """
    Calculates the Jacobian matrix (J) of the DFMI model.
    (This function remains unchanged from the previous version)
    """
    a, m, phi, psi = param
    J = np.zeros((2 * ndata, 4))
    j = np.arange(1, ndata + 1)
    
    phase_term = np.cos(phi + j * np.pi / 2.0)
    cos_jpsi = np.cos(j * psi)
    sin_jpsi = np.sin(j * psi)
    
    bessel_j = jv(j, m)
    bessel_deriv = 0.5 * (jv(j - 1, m) - jv(j + 1, m))
    
    common_term = a * phase_term * bessel_j
    model_q = common_term * cos_jpsi
    model_i = -common_term * sin_jpsi
    
    if a != 0:
        J[:ndata, 0] = model_q / a
        J[ndata:, 0] = model_i / a

    common_deriv_term_m = a * phase_term * bessel_deriv
    J[:ndata, 1] = common_deriv_term_m * cos_jpsi
    J[ndata:, 1] = -common_deriv_term_m * sin_jpsi

    phase_deriv_term = np.cos(phi + j * np.pi / 2.0 + np.pi / 2.0)
    common_deriv_term_phi = a * phase_deriv_term * bessel_j
    J[:ndata, 2] = common_deriv_term_phi * cos_jpsi
    J[ndata:, 2] = -common_deriv_term_phi * sin_jpsi

    J[:ndata, 3] = common_term * -sin_jpsi * j
    J[ndata:, 3] = -common_term * cos_jpsi * j
    
    return J

def calculate_m_precision(m_range, ndata, snr_db):
    """
    Core calculation function for statistical uncertainty of 'm'.
    
    Parameters
    ----------
    m_range : array_like
        The range of modulation depths 'm' to analyze.
    ndata : int
        Number of harmonics to use in the fit.
    snr_db : float
        Signal-to-Noise Ratio in dB for the I/Q measurements.

    Returns
    -------
    numpy.ndarray
        An array of the statistical uncertainty (delta_m) for each m in m_range.
    """
    param_fixed = np.array([1.0, 0, np.pi/4, 0.0]) # a, m, phi, psi
    snr_linear = 10**(snr_db / 20.0)
    noise_variance = (1.0 / snr_linear)**2
    
    delta_m_list = []
    for m_true in m_range:
        param_fixed[1] = m_true
        J = calculate_jacobian(ndata, param_fixed)
        JTJ = J.T @ J
        try:
            covariance_matrix = noise_variance * inv(JTJ)
            delta_m = np.sqrt(covariance_matrix[1, 1])
            delta_m_list.append(delta_m)
        except np.linalg.LinAlgError:
            delta_m_list.append(np.inf)
            
    return np.array(delta_m_list)
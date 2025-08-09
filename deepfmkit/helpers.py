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
import numpy as np
from scipy.special import jv
from scipy.linalg import inv


def calculate_jacobian(ndata: int, param: np.ndarray) -> np.ndarray:
    """Calculates the Jacobian matrix (J) of the ideal DFMI model.

    The Jacobian describes the sensitivity of the model's outputs (the I/Q
    values of each harmonic) to changes in its input parameters. It is a
    matrix of shape `(2 * ndata, 4)`.

    Parameters
    ----------
    ndata : int
        The number of harmonics (from 1 to ndata) to include in the model.
    param : np.ndarray
        A 4-element array containing the model parameters: `[amp, m, phi, psi]`.

    Returns
    -------
    np.ndarray
        The Jacobian matrix `J`, where `J[i, j]` is the partial derivative of
        the i-th model output with respect to the j-th parameter. The first
        `ndata` rows correspond to the Q components, and the next `ndata` rows
        correspond to the I components.
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


def calculate_m_precision(m_range: np.ndarray, ndata: int, snr_db: float) -> np.ndarray:
    """Calculates the statistical uncertainty of the 'm' parameter.

    This function computes the theoretical precision (standard deviation) with
    which the modulation depth `m` can be estimated, based on the Cramer-Rao
    Lower Bound (CRLB). It assumes an idealized NLS fit where noise on the
    I/Q measurements is the only source of error.

    Parameters
    ----------
    m_range : np.ndarray
        An array of modulation depth `m` values (in radians) to analyze.
    ndata : int
        The number of harmonics to use in the fit.
    snr_db : float
        The Signal-to-Noise Ratio in dB for the I/Q measurements. This SNR
        is defined relative to the signal amplitude `amp=1`.

    Returns
    -------
    np.ndarray
        An array of the calculated statistical uncertainty (delta_m), with
        the same shape as `m_range`. Values will be `np.inf` if the
        calculation fails (e.g., due to a singular matrix).

    Notes
    -----
    The calculation relies on the Fisher Information Matrix, which is
    approximated by `J^T * J`. The covariance matrix of the estimated
    parameters is the inverse of the Fisher matrix, scaled by the noise
    variance. The uncertainty in `m` is the square root of the corresponding
    diagonal element of the covariance matrix.
    """
    param_fixed = np.array([1.0, 0, np.pi / 4, 0.0])  # a, m, phi, psi
    snr_linear = 10 ** (snr_db / 20.0)
    noise_variance = (1.0 / snr_linear) ** 2

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

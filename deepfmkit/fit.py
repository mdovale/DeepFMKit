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
"""Low-level implementation of the Non-Linear Least Squares (NLS) fitter.

This module contains the core, performance-critical functions for the
Levenberg-Marquardt Algorithm (LMA) tailored to the DFMI signal model. The
functions are heavily optimized using NumPy vectorization to avoid slow Python
loops and achieve high throughput.

The main entry point is the `fit()` function, which orchestrates the LMA loop
and includes robust fallback mechanisms for improved convergence.

Constants
---------
NPARMS : int
    The total number of parameters in the full DFMI model (amp, m, phi, psi).
MAX_LMA_STEPS : int
    The maximum number of iterations for the Levenberg-Marquardt loop.
LMA_CONVERGENCE_IMPROVE : float
    The minimum relative improvement in Sum-of-Squares (SSQ) required to
    continue the LMA loop.
LMA_CONVERGENCE_PARAM_CHANGE : float
    The minimum relative change in the parameter vector norm required to
    continue the LMA loop.
FITOK_THRESHOLD : float
    An SSQ value below which a fit is considered "good", skipping
    the retry mechanism.
"""

import numpy as np
from math import cos, sin
from scipy.special import jv
from typing import List, Tuple

NPARMS = 4
MAX_LMA_STEPS = 100
LMA_CONVERGENCE_IMPROVE = 1e-9
LMA_CONVERGENCE_PARAM_CHANGE = 1e-9
FITOK_THRESHOLD = 1e-3

# Grid search parameters for finding a better initial guess
M_GRID_MIN = 3.0
M_GRID_MAX = 30.0
M_GRID_STEP = 0.5
BESSEL_AMP_THRESHOLD = 0.05
SINCOS_AMP_THRESHOLD = 0.1
# Define the full parameter set for reference
ALL_PARAMS = ["amp", "m", "phi", "psi"]
DEFAULT_GUESS = {"amp": 1.6, "m": 6.0, "phi": 0.0, "psi": 0.0}


def calculate_quadratures(
    n: int, data: np.ndarray, w0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the in-phase (I) and quadrature (Q) components of a signal.

    This function performs digital lock-in amplification by demodulating the
    input `data` with reference sine and cosine waves at a specific harmonic
    of the fundamental frequency. The mean of the returned arrays gives the
    final I/Q value for the specified harmonic over the entire data buffer.

    Parameters
    ----------
    n : int
        The zero-indexed harmonic number to demodulate. The demodulation
        frequency will be `(n + 1) * fm`.
    data : np.ndarray
        A 1D array of the raw time-series signal to be demodulated.
    w0 : float
        The fundamental angular frequency of the modulation in units of
        radians per sample (i.e., `2 * pi * fm / f_samp`).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - Q_data: The instantaneous in-phase (cosine) component.
        - I_data: The instantaneous quadrature (sine) component.
    """
    # Ensure data is a NumPy array for efficient operations
    data = np.asarray(data)

    # Generate the time steps for the entire buffer at once
    # t_steps is an array [0, 1, 2, ..., bufferSize-1]
    t_steps = np.arange(len(data))

    # Calculate the argument for the sine and cosine functions for all time steps
    # This is also a vectorized operation.
    demod_angle = (n + 1) * w0 * t_steps

    # Perform element-wise multiplication on the entire arrays.
    # This is a single, highly optimized operation in NumPy.
    Q_data = data * np.cos(demod_angle)
    I_data = data * np.sin(demod_angle)

    return Q_data, I_data


def coeffs(
    ndata: int, data: np.ndarray, param: np.ndarray, fit_params: List[str] = ALL_PARAMS
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calculates the SSQ, Jacobian (J^T*J), and gradient (J^T*r).

    This is a fully vectorized, high-performance implementation that is the
    core of the LMA loop. It computes the sum-of-squared-residuals (ssq),
    the approximate Hessian matrix (`JTJ`), and the gradient vector for a
    given set of parameters.

    Parameters
    ----------
    ndata : int
        Number of harmonics to use in the fit.
    data : np.ndarray
        A 1D array of size `2*ndata` containing the measured I/Q values
        ([Q1..QN, I1..IN]).
    param : np.ndarray
        A vector containing the current guess for the parameters being
        actively fitted. Its length must match `len(fit_params)`.
    fit_params : list[str], optional
        A list of parameter names being actively fitted. Defaults to all four.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        A tuple containing:
        - ssq: The sum of squared residuals.
        - JTJ_flat: The flattened (`n_params` * `n_params`) array of the
          Jacobian matrix (`J^T * J`).
        - gradient: The (`n_params`,) gradient vector (`J^T * r`).
    """
    # Create a full parameter vector, filling in fixed values.
    # We assume fixed values are the standard initial guesses or 0.
    full_param = DEFAULT_GUESS.copy()
    for i, p_name in enumerate(fit_params):
        full_param[p_name] = param[i]

    a, m, phi, psi = (
        full_param["amp"],
        full_param["m"],
        full_param["phi"],
        full_param["psi"],
    )

    # --- 1. Prepare harmonic-dependent arrays ---
    j = np.arange(1, ndata + 1)  # Harmonic orders: [1, 2, ..., ndata]

    # This term, cos(phi + j*pi/2), is the core of the model's structure.
    # It correctly captures the alternating sin/cos and sign changes.
    phase_term = np.cos(phi + j * np.pi / 2.0)

    cos_jpsi = np.cos(j * psi)
    sin_jpsi = np.sin(j * psi)

    # --- 2. Vectorized Bessel function calculation (one call) ---
    bessel_j = jv(j, m)
    # The derivative of J_n(x) is 0.5 * (J_{n-1}(x) - J_{n+1}(x))
    bessel_deriv = 0.5 * (jv(j - 1, m) - jv(j + 1, m))

    # --- 3. Calculate the model values for all harmonics at once ---
    common_term = a * phase_term * bessel_j
    model_q = common_term * cos_jpsi
    model_i = common_term * sin_jpsi
    model_i = -common_term * sin_jpsi

    # --- 4. Calculate residuals and SSQ ---
    q_data = data[:ndata]
    i_data = data[ndata:]
    residuals = np.concatenate([q_data - model_q, i_data - model_i])
    ssq = np.dot(residuals, residuals)

    # --- 5. Calculate the full (2*ndata, 4) Jacobian matrix J ---
    n_active_params = len(fit_params)
    J = np.zeros((2 * ndata, n_active_params))

    # Loop through the active parameters and calculate their derivatives
    for i, p_name in enumerate(fit_params):
        if p_name == "amp":
            if a != 0:
                J[:ndata, i] = model_q / a
                J[ndata:, i] = model_i / a
        elif p_name == "m":
            common_deriv_term_m = a * phase_term * bessel_deriv
            J[:ndata, i] = common_deriv_term_m * cos_jpsi
            J[ndata:, i] = -common_deriv_term_m * sin_jpsi
        elif p_name == "phi":
            phase_deriv_term = np.cos(phi + j * np.pi / 2.0 + np.pi / 2.0)
            common_deriv_term_phi = a * phase_deriv_term * bessel_j
            J[:ndata, i] = common_deriv_term_phi * cos_jpsi
            J[ndata:, i] = -common_deriv_term_phi * sin_jpsi
        elif p_name == "psi":
            J[:ndata, i] = common_term * -sin_jpsi * j
            J[ndata:, i] = -common_term * cos_jpsi * j

    # --- Calculate final matrices (now correctly sized) ---
    JTJ = J.T @ J
    gradient = J.T @ residuals

    return ssq, JTJ.flatten(), gradient


def ssqf(
    ndata: int, data: np.ndarray, param: np.ndarray, fit_params: List[str] = ALL_PARAMS
) -> float:
    """A minimal, fast, vectorized function to calculate only the SSQ.

    This lightweight version is used within the LMA loop when testing different
    damping parameters, as it avoids the expensive calculation of the Jacobian.

    Parameters are identical to `coeffs`.

    Returns
    -------
    float
        The sum of squared residuals for the given parameters.
    """
    full_param = DEFAULT_GUESS.copy()
    for i, p_name in enumerate(fit_params):
        full_param[p_name] = param[i]

    a, m, phi, psi = (
        full_param["amp"],
        full_param["m"],
        full_param["phi"],
        full_param["psi"],
    )

    j = np.arange(1, ndata + 1)

    phase_term = np.cos(phi + j * np.pi / 2.0)
    bessel_j = jv(j, m)
    common_term = a * phase_term * bessel_j

    model_q = common_term * np.cos(j * psi)
    model_i = -common_term * np.sin(j * psi)

    residuals = np.concatenate([data[:ndata] - model_q, data[ndata:] - model_i])
    return np.dot(residuals, residuals)


def msolve(lam: float, a_g_mat: np.ndarray, b_g_mat: np.ndarray) -> np.ndarray:
    """Solves the Levenberg-Marquardt matrix equation for the parameter update step.

    This function calculates the parameter update `dp` by solving the equation:
    `(J^T*J + lam * diag(J^T*J)) * dp = J^T*r`

    Parameters
    ----------
    lam : float
        The Levenberg-Marquardt damping parameter.
    a_g_mat : np.ndarray
        A flattened array representing the `J^T*J` matrix.
    b_g_mat : np.ndarray
        An array representing the gradient vector `J^T*r`.

    Returns
    -------
    np.ndarray
        An array `dp` containing the calculated parameter update step.
        Returns a zero vector if the matrix is singular.
    """
    n_params = len(b_g_mat)

    # Reshape the flattened array into a 4x4 matrix
    JTJ = a_g_mat.reshape(n_params, n_params)

    # Create the damped matrix for the L-M algorithm
    # This uses the variant with lambda on the diagonal, which is more stable.
    A_lm = JTJ + lam * np.diag(np.diag(JTJ))

    try:
        # Use NumPy's highly optimized and stable linear algebra solver
        dp = np.linalg.solve(A_lm, b_g_mat)
    except np.linalg.LinAlgError:
        # If the matrix is singular (e.g., m=0), the problem is ill-defined.
        # The best thing to do is propose no change to the parameters.
        # The L-M algorithm will then increase lambda and try again.
        dp = np.zeros(n_params)

    return dp


def _run_lma_fit(
    ndata: int,
    data: np.ndarray,
    initial_guess: np.ndarray,
    fit_params: List[str] = ALL_PARAMS,
) -> Tuple[np.ndarray, float]:
    """Executes the core Levenberg-Marquardt loop until convergence.

    This internal helper function implements the iterative LMA procedure.
    """
    parm = initial_guess.copy()

    # Calculate initial state
    ssq0, JTJ_flat, gradient = coeffs(ndata, data, parm, fit_params)

    for _ in range(MAX_LMA_STEPS):
        parm_old = parm.copy()

        # Robustly find a damping factor that improves the fit
        lambda_candidates = [0.0, 1e-7, 1e-5, 1e-3, 1e-1, 1, 10, 100]

        # Start by assuming no improvement is found
        best_ssq_step = ssq0
        best_parm_step = parm

        for lam in lambda_candidates:
            dp = msolve(lam, JTJ_flat, gradient)
            if np.linalg.norm(dp) < 1e-15:
                continue

            p_try = parm + dp

            # Calculate the SSQ for the trial parameters *without* re-calculating the Jacobian.
            # We can do this with a simplified, fast ssq-only function or by inlining it.
            # Let's create a minimal, vectorized ssq function.
            ssq_try = ssqf(ndata, data, p_try, fit_params)

            if ssq_try < best_ssq_step:
                best_ssq_step = ssq_try
                best_parm_step = p_try
                break  # Found an improvement, take the step

        # If no improvement was found, exit the loop
        if best_ssq_step >= ssq0:
            break

        # An improvement was found, update parameters and calculate new state
        parm = best_parm_step
        ssq0, JTJ_flat, gradient = coeffs(ndata, data, parm, fit_params)

        # Check for convergence
        param_change = np.linalg.norm(np.array(parm) - np.array(parm_old))
        if (
            ssq0 - best_ssq_step
        ) < LMA_CONVERGENCE_IMPROVE and param_change < LMA_CONVERGENCE_PARAM_CHANGE:
            break

    return parm, ssq0  # Return the final converged state


def _find_best_initial_guess(ndata: int, data: np.ndarray) -> np.ndarray:
    """Performs a grid search over 'm' to find a better initial guess.

    This robust fallback routine is triggered when the default guess fails.
    It performs a coarse grid search over the modulation depth `m` and, for
    each `m`, analytically estimates the optimal amplitude and phase to

    find a promising region of the parameter space to start a new fit.
    """
    best_ssq_guess = 9e99
    best_parm_guess = np.zeros(NPARMS)

    # This logic is a direct translation of the original retry mechanism
    psitry = 0.0  # Assuming psi is stable and near zero
    for mtry in np.arange(M_GRID_MIN, M_GRID_MAX + M_GRID_STEP, M_GRID_STEP):
        sinsum, cossum = 0.0, 0.0
        nsin, ncos = 0, 0
        j = np.arange(1, ndata + 1)

        bes_q = jv(j, mtry) * np.cos(j * psitry)
        bes_i = jv(j, mtry) * -np.sin(j * psitry)

        q_data = data[:ndata]
        i_data = data[ndata:]

        # Linear estimate of phi and amplitude based on this mtry
        # This part is complex and highly specific to the original code's logic.
        # It could be simplified, but for now we translate it directly.
        for i in range(ndata):
            if abs(bes_q[i]) > BESSEL_AMP_THRESHOLD:
                if j[i] % 4 == 0:
                    cossum += q_data[i] / bes_q[i]
                    ncos += 1
                elif j[i] % 4 == 1:
                    sinsum -= q_data[i] / bes_q[i]
                    nsin += 1
                elif j[i] % 4 == 2:
                    cossum -= q_data[i] / bes_q[i]
                    ncos += 1
                elif j[i] % 4 == 3:
                    sinsum += q_data[i] / bes_q[i]
                    nsin += 1
            if abs(bes_i[i]) > BESSEL_AMP_THRESHOLD:
                if j[i] % 4 == 0:
                    cossum += i_data[i] / bes_i[i]
                    ncos += 1
                elif j[i] % 4 == 1:
                    sinsum -= i_data[i] / bes_i[i]
                    nsin += 1
                elif j[i] % 4 == 2:
                    cossum -= i_data[i] / bes_i[i]
                    ncos += 1
                elif j[i] % 4 == 3:
                    sinsum += i_data[i] / bes_i[i]
                    nsin += 1

        if nsin == 0 or ncos == 0:
            continue
        ptry = np.arctan2(sinsum / nsin, cossum / ncos)

        asum, na = 0.0, 0
        sincos_table = np.array([cos(ptry), -sin(ptry), -cos(ptry), sin(ptry)])

        for i in range(ndata):
            if (
                abs(bes_q[i]) > BESSEL_AMP_THRESHOLD
                and abs(sincos_table[j[i] % 4]) > SINCOS_AMP_THRESHOLD
            ):
                asum += q_data[i] / (sincos_table[j[i] % 4] * bes_q[i])
                na += 1
            if (
                abs(bes_i[i]) > BESSEL_AMP_THRESHOLD
                and abs(sincos_table[j[i] % 4]) > SINCOS_AMP_THRESHOLD
            ):
                asum += i_data[i] / (sincos_table[j[i] % 4] * bes_i[i])
                na += 1

        if na == 0:
            continue
        atry = asum / na

        parm_try = np.array([atry, mtry, ptry, psitry])
        ssq_try = ssqf(ndata, data, parm_try)

        if ssq_try < best_ssq_guess:
            best_ssq_guess = ssq_try
            best_parm_guess = parm_try

    return best_parm_guess


def fit(
    ndata: int, data: np.ndarray, parm: np.ndarray, fit_params: List[str] = ALL_PARAMS
) -> Tuple[int, np.ndarray, float]:
    """Main entry point for the NLS fitting algorithm for a single data buffer.

    This function attempts a fit using the provided initial guess. If the fit
    quality is poor and all four parameters are being fitted, it triggers a
    grid search to find a better starting point and retries the fit.

    Parameters
    ----------
    ndata : int
        Number of harmonics to use in the fit.
    data : np.ndarray
        A 1D array of size `2*ndata` containing the measured I/Q values.
    parm : np.ndarray
        A vector containing the initial guess for the parameters being
        actively fitted.
    fit_params : list[str], optional
        A list of parameter names being actively fitted.

    Returns
    -------
    tuple[int, np.ndarray, float]
        A tuple containing:
        - status: An integer code (0: good fit, 1: good after retry, 2: bad fit).
        - final_parm_vector: The final 4-element parameter vector `[amp, m, phi, psi]`.
        - fit_ssq: The final sum of squared residuals.
    """
    # --- First, try fitting from the provided initial guess ---
    # The _run_lma_fit helper is now flexible and handles any `fit_params`.
    fit_parm_active, fit_ssq = _run_lma_fit(ndata, data, parm, fit_params)

    # Check if the fit is good enough. If so, we are likely done.
    if fit_ssq < FITOK_THRESHOLD:
        status = 0  # Good fit on first try
    else:
        # If the fit is poor, decide whether to retry.
        # We only use the complex retry logic for the difficult, unconstrained 4-parameter case.
        if sorted(fit_params) == sorted(ALL_PARAMS):
            # Try to find a better 4-parameter starting point and refit.
            best_guess_parm = _find_best_initial_guess(ndata, data)

            if np.any(best_guess_parm):  # If the grid search found a valid alternative
                # The guess from the grid search is a full 4-param vector.
                # `parm` here is also the full 4-param vector from the first attempt.
                fit_parm_retry, fit_ssq_retry = _run_lma_fit(
                    ndata, data, best_guess_parm, fit_params
                )

                # Use the result of the retry if it was better
                if fit_ssq_retry < fit_ssq:
                    fit_parm_active = fit_parm_retry
                    fit_ssq = fit_ssq_retry

            status = (
                1 if fit_ssq < FITOK_THRESHOLD else 2
            )  # Good after retry, or still bad
        else:
            # For constrained fits, a poor result is just a poor result. No retry.
            status = 2  # Still bad

    # --- Reconstruct the full 4-parameter vector from the active ones ---
    # This logic correctly handles both constrained and unconstrained cases.
    final_parm_dict = DEFAULT_GUESS.copy()
    # For a 4-param fit, fit_parm_active contains all 4. For a 3-param fit, it contains 3.
    for i, p_name in enumerate(fit_params):
        final_parm_dict[p_name] = fit_parm_active[i]

    # Wrap phi to [-pi,pi]
    final_parm_dict["phi"] = (final_parm_dict["phi"] + np.pi) % (2 * np.pi) - np.pi

    # Convert dict back to a standard 4-element numpy array for the return
    final_parm_vector = np.array(
        [
            final_parm_dict["amp"],
            final_parm_dict["m"],
            final_parm_dict["phi"],
            final_parm_dict["psi"],
        ]
    )

    return status, final_parm_vector, fit_ssq

import numpy as np
from math import cos, sin, log, sqrt, exp
from scipy.special import jv
from scipy.optimize import brent

NPARMS = 4
MAXDATA = 40 # default 20
MAX_LMA_STEPS = 100
LMA_CONVERGENCE_IMPROVE = 1e-9  # Min improvement in SSQ to continue
LMA_CONVERGENCE_PARAM_CHANGE = 1e-9 # Min change in parameter vector norm
FITOK_THRESHOLD = 1e-3
# Grid search parameters for finding a better initial guess
M_GRID_MIN = 5.0
M_GRID_MAX = 30.0
M_GRID_STEP = 0.5
BESSEL_AMP_THRESHOLD = 0.05
SINCOS_AMP_THRESHOLD = 0.1

def calculate_quadratures(n, data, w0):
    """Calculate the in-phase (Q) and quadrature (I) components of a signal.

    This function performs digital lock-in amplification by demodulating the
    input `data` with reference sine and cosine waves at a specific harmonic
    of the fundamental frequency.

    Parameters
    ----------
    n : int
        The zero-indexed harmonic number to demodulate. The demodulation
        frequency will be `(n + 1) * f_mod`.
    data : array_like
        A 1-D array of the raw time-series signal to be demodulated.
    w0 : float
        The fundamental angular frequency of the modulation in units of
        radians per sample (i.e., `2 * pi * f_mod / f_samp`).

    Returns
    -------
    Q_data : numpy.ndarray
        A 1-D array of the same size as `data`, containing the in-phase (cosine)
        component of the demodulated signal at each time step.
    I_data : numpy.ndarray
        A 1-D array of the same size as `data`, containing the quadrature (sine)
        component of the demodulated signal at each time step.

    Notes
    -----
    The mean of the returned Q and I arrays gives the final I/Q value 
    for the specified harmonic over the buffer.
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

def coeffs(ndata, data, param):
    """
    Calculate the sum-of-squares, Jacobian (J^T*J), and gradient (J^T*r).

    This is a fully vectorized implementation that is mathematically identical
    to the original looped `coeffs` function for high performance.

    Parameters
    ----------
    ndata : int
        Number of harmonics to use in the fit.
    data : numpy.ndarray
        A 1D array of size 2*ndata containing the measured I/Q values (Q then I).
    param : numpy.ndarray
        A 4-element array with the current parameter guess: [a, m, phi, psi].

    Returns
    -------
    ssq : float
        The sum of squared residuals for the given parameters.
    JTJ_flat : numpy.ndarray
        The flattened 16-element array of the 4x4 Jacobian matrix (J^T * J).
    gradient : numpy.ndarray
        The 4-element gradient vector (J^T * r).
    """
    a, m, phi, psi = param

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
    J = np.zeros((2 * ndata, 4))

    # Column 0: Derivative w.r.t. 'a' (amplitude)
    if a != 0:
        J[:ndata, 0] = model_q / a
        J[ndata:, 0] = model_i / a

    # Column 1: Derivative w.r.t. 'm' (modulation depth)
    common_deriv_term_m = a * phase_term * bessel_deriv
    J[:ndata, 1] = common_deriv_term_m * cos_jpsi
    J[ndata:, 1] = -common_deriv_term_m * sin_jpsi

    # Column 2: Derivative w.r.t. 'phi' (interferometric phase)
    # d/dphi(cos(phi + C)) = -sin(phi + C) = cos(phi + C + pi/2)
    phase_deriv_term = np.cos(phi + j * np.pi / 2.0 + np.pi / 2.0)
    common_deriv_term_phi = a * phase_deriv_term * bessel_j
    J[:ndata, 2] = common_deriv_term_phi * cos_jpsi
    J[ndata:, 2] = -common_deriv_term_phi * sin_jpsi

    # Column 3: Derivative w.r.t. 'psi' (modulation phase)
    J[:ndata, 3] = common_term * -sin_jpsi * j
    J[ndata:, 3] = -common_term * cos_jpsi * j
    
    # --- 6. Calculate final matrices using fast matrix multiplication ---
    JTJ = J.T @ J
    gradient = J.T @ residuals

    return ssq, JTJ.flatten(), gradient

def ssqf(ndata, data, param):
    """
    A minimal, fast, vectorized function to calculate only the SSQ.
    """
    a, m, phi, psi = param
    j = np.arange(1, ndata + 1)
    
    phase_term = np.cos(phi + j * np.pi / 2.0)
    bessel_j = jv(j, m)
    common_term = a * phase_term * bessel_j
    
    model_q = common_term * np.cos(j * psi)
    model_i = -common_term * np.sin(j * psi)
    
    residuals = np.concatenate([data[:ndata] - model_q, data[ndata:] - model_i])
    return np.dot(residuals, residuals)

def msolve(lam, a_g_mat, b_g_mat):
    """Solves the Levenberg-Marquardt matrix equation.

    Calculates the parameter update step dp by solving the equation:
    (J^T*J + lam * diag(J^T*J)) * dp = J^T*r

    Parameters
    ----------
    lam : float
        The Levenberg-Marquardt damping parameter.
    a_g_mat : numpy.ndarray
        A flattened 16-element array representing the 4x4 (J^T*J) matrix.
    b_g_mat : numpy.ndarray
        A 4-element array representing the gradient vector (J^T*r).

    Returns
    -------
    numpy.ndarray
        A 4-element array `dp` containing the calculated parameter update step.
        Returns a zero vector if the matrix is singular.
    """
    # Reshape the flattened array into a 4x4 matrix
    JTJ = a_g_mat.reshape(4, 4)

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
        dp = np.zeros(NPARMS)
        
    return dp

def _run_lma_fit(ndata, data, initial_guess):
    """
    Executes the core Levenberg-Marquardt loop until convergence.
    This version avoids redundant SSQ calculations by only calling coeffs.
    """
    parm = initial_guess.copy()
    
    # Calculate initial state
    ssq0, JTJ_flat, gradient = coeffs(ndata, data, parm)

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
            ssq_try = ssqf(ndata, data, p_try) # New function needed!
            
            if ssq_try < best_ssq_step:
                best_ssq_step = ssq_try
                best_parm_step = p_try
                break # Found an improvement, take the step
        
        # If no improvement was found, exit the loop
        if best_ssq_step >= ssq0:
            break

        # An improvement was found, update parameters and calculate new state
        parm = best_parm_step
        ssq0, JTJ_flat, gradient = coeffs(ndata, data, parm)

        # Check for convergence
        param_change = np.linalg.norm(parm - parm_old)
        if (ssq0 - best_ssq_step) < LMA_CONVERGENCE_IMPROVE and param_change < LMA_CONVERGENCE_PARAM_CHANGE:
            break
            
    return parm, ssq0 # Return the final converged state

def _find_best_initial_guess(ndata, data):
    """
    Performs a grid search over 'm' to find a promising initial guess
    for the NLS fit when the default guess fails.
    """
    best_ssq_guess = 9e99
    best_parm_guess = np.zeros(NPARMS)
    
    # This logic is a direct translation of the original retry mechanism
    psitry = 0.0 # Assuming psi is stable and near zero
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
                if j[i] % 4 == 0: cossum += q_data[i] / bes_q[i]; ncos += 1
                elif j[i] % 4 == 1: sinsum -= q_data[i] / bes_q[i]; nsin += 1
                elif j[i] % 4 == 2: cossum -= q_data[i] / bes_q[i]; ncos += 1
                elif j[i] % 4 == 3: sinsum += q_data[i] / bes_q[i]; nsin += 1
            if abs(bes_i[i]) > BESSEL_AMP_THRESHOLD:
                if j[i] % 4 == 0: cossum += i_data[i] / bes_i[i]; ncos += 1
                elif j[i] % 4 == 1: sinsum -= i_data[i] / bes_i[i]; nsin += 1
                elif j[i] % 4 == 2: cossum -= i_data[i] / bes_i[i]; ncos += 1
                elif j[i] % 4 == 3: sinsum += i_data[i] / bes_i[i]; nsin += 1

        if nsin == 0 or ncos == 0: continue
        ptry = np.arctan2(sinsum / nsin, cossum / ncos)

        asum, na = 0.0, 0
        sincos_table = np.array([cos(ptry), -sin(ptry), -cos(ptry), sin(ptry)])
        
        for i in range(ndata):
            if abs(bes_q[i]) > BESSEL_AMP_THRESHOLD and abs(sincos_table[j[i]%4]) > SINCOS_AMP_THRESHOLD:
                asum += q_data[i] / (sincos_table[j[i]%4] * bes_q[i])
                na += 1
            if abs(bes_i[i]) > BESSEL_AMP_THRESHOLD and abs(sincos_table[j[i]%4]) > SINCOS_AMP_THRESHOLD:
                asum += i_data[i] / (sincos_table[j[i]%4] * bes_i[i])
                na += 1

        if na == 0: continue
        atry = asum / na
        
        parm_try = np.array([atry, mtry, ptry, psitry])
        ssq_try = ssqf(ndata, data, parm_try)

        if ssq_try < best_ssq_guess:
            best_ssq_guess = ssq_try
            best_parm_guess = parm_try

    return best_parm_guess

def fit(ndata, data, parm):
    """
    Main entry point for the NLS fitting algorithm.

    This function first attempts to find a solution using the provided initial
    parameter guess. If the resulting fit quality (SSQ) is poor, it triggers
    a grid search to find a better starting point and retries the fit.
    """
    # First, try fitting from the provided initial guess
    fit_parm, fit_ssq = _run_lma_fit(ndata, data, parm)
    
    # Check if the fit is good enough
    if fit_ssq < FITOK_THRESHOLD:
        status = 0 # Good fit on first try
    else:
        # If not, try to find a better starting point and refit
        best_guess_parm = _find_best_initial_guess(ndata, data)
        
        # Check if the grid search found a potentially better guess
        if np.any(best_guess_parm): # If it's not all zeros
            fit_parm_retry, fit_ssq_retry = _run_lma_fit(ndata, data, best_guess_parm)
            
            # Use the result of the retry if it was better
            if fit_ssq_retry < fit_ssq:
                fit_parm = fit_parm_retry
                fit_ssq = fit_ssq_retry

        status = 1 if fit_ssq < FITOK_THRESHOLD else 2 # Good after retry, or still bad

    # Final parameter normalization (ensures a>0, m>0)
    if fit_parm[0] < 0:
        fit_parm[0] *= -1
        fit_parm[2] += np.pi
    if fit_parm[1] < 0:
        fit_parm[1] *= -1
        fit_parm[2] += np.pi
        
    # Ensure phi is wrapped to (-pi, pi]
    fit_parm[2] = (fit_parm[2] + np.pi) % (2 * np.pi) - np.pi

    return status, fit_parm, fit_ssq
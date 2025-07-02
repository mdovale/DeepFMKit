try:
    profile
except NameError:
    def profile(func):
        return func

from .goto import with_goto

import numpy as np
from math import cos, sin, log, sqrt, exp
from scipy.special import jv
from scipy.optimize import brent

NPARMS = 4
MAXDATA = 40 # default 20
LAMBDA_MIN = 1e-10 # default 1e-10
LAMBDA_MAX = 1e6 # default 1e6
BRENT_TOL = 1e-2 # default 1e-2
MAR_END = 1e-15  # default 1e-15
# Difference between ssq of parameters and next parameters, that will make fit stop
MMIN = 5 # default 5
MMAX = 30 # default 30
MSTEP = 0.25 # default 0.25
FITOK = 1e-3 # default 1e-3
# Denotes the value of ssq (sum of squares), which is low enough to accept the fit 
# without searching for other starting values
BESLIM = 0.05 # default 0.05
SINLIM = 0.1 # default 0.1
DBL_EPSILON = 2.2204460492503131e-16


@profile
def calculate_quadratures(n, data, w0, bufferSize):
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
    bufferSize : int
        The number of samples in the `data` array.

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
    This implementation is vectorized using NumPy for high performance,
    avoiding explicit Python loops. The mean of the returned Q and I arrays
    gives the final I/Q value for the specified harmonic over the buffer.
    """
    # Ensure data is a NumPy array for efficient operations
    data = np.asarray(data)

    # Generate the time steps for the entire buffer at once
    # t_steps is an array [0, 1, 2, ..., bufferSize-1]
    t_steps = np.arange(bufferSize)

    # Calculate the argument for the sine and cosine functions for all time steps
    # This is also a vectorized operation.
    demod_angle = (n + 1) * w0 * t_steps

    # Perform element-wise multiplication on the entire arrays.
    # This is a single, highly optimized operation in NumPy.
    Q_data = data * np.cos(demod_angle)
    I_data = data * np.sin(demod_angle)

    return Q_data, I_data

@profile
def mean_filter(signal):
	return signal.mean()

@profile
# In fit.py

def coeffs(ndata, data, param):
    """
    Calculates the sum of squares residual, the Jacobian matrix (J^T*J),
    and the gradient vector (J^T*r) for the NLS fit.
    """
    a = param[0]
    m = param[1]
    phi = param[2]
    psi = param[3]

    a_g_mat = np.zeros(16)
    b_g_mat = np.zeros(4)
    sincos = np.zeros(4)

    sincos[0] = cos(phi)
    sincos[1] = -sin(phi)
    sincos[2] = -cos(phi)
    sincos[3] = sin(phi)

    ssq = 0.0

    # Pre-calculate all model values for printing
    model_q = np.zeros(ndata)
    model_i = np.zeros(ndata)
    
    for i in range(ndata):
        j = i + 1
        model_q[i] = a * sincos[j % 4] * jv(j, m) * cos(j * psi)
        
    for i in range(ndata, 2*ndata):
        j = (i - ndata) + 1
        model_i[j-1] = a * sincos[j % 4] * jv(j, m) * (-1) * sin(j * psi)
    
    # ===================================================================
    # START: Diagnostic Print Block
    # ===================================================================
    # print("\n" + "="*80)
    # print("--- DIAGNOSTIC: Entering coeffs function ---")
    # print(f"Parameters (a, m, phi, psi): {a:.4f}, {m:.4f}, {phi:.4f}, {psi:.4f}")
    # print("\nComparing Measured Data vs. Calculated Model:")
    # print("--------------------------------------------------------------------")
    # print(" n |  Measured Q  |   Model Q    |  Measured I  |   Model I")
    # print("--------------------------------------------------------------------")
    # for i in range(ndata):
    #     # data[i] is Q_n, data[i+ndata] is I_n
    #     print(f"{i+1:2d} | {data[i]:+1.8f} | {model_q[i]:+1.8f} | {data[i+ndata]:+1.8f} | {model_i[i]:+1.8f}")
    # print("--------------------------------------------------------------------")
    # ===================================================================
    # END: Diagnostic Print Block
    # ===================================================================

    for i in range(ndata):
        j = i+1

        d0 = sincos[j % 4] * jv(j, m) * cos(j * psi)
        d1 = a * sincos[j % 4] * 0.5 * (jv(j-1, m) - jv(j+1, m)) * cos(j * psi)
        d2 = a * sincos[(j + 1) % 4] * jv(j, m) * cos(j * psi)
        d3 = - a * sincos[j % 4] * jv(j, m) * j * sin(j * psi)

        # Use the pre-calculated model value for Q_n
        ydiff = data[i] - model_q[i]

        b_g_mat[0]  += ydiff * d0
        b_g_mat[1]  += ydiff * d1
        b_g_mat[2]  += ydiff * d2
        b_g_mat[3]  += ydiff * d3
        a_g_mat[0]  += d0 * d0 #a11
        a_g_mat[1]  += d0 * d1 #a12
        a_g_mat[2]  += d0 * d2 #a13
        a_g_mat[3]  += d0 * d3 #a14
        a_g_mat[5]  += d1 * d1 #a22
        a_g_mat[6]  += d1 * d2 #a23
        a_g_mat[7]  += d1 * d3 #a24
        a_g_mat[10] += d2 * d2 #a33
        a_g_mat[11] += d2 * d3 #a34
        a_g_mat[15] += d3 * d3 #a44
        
        ssq += ydiff**2

    for i in range(ndata, 2*ndata):
        # Index for model_i is (0 to ndata-1)
        model_idx = i - ndata
        j = model_idx + 1

        d0 = - sincos[j % 4] * jv(j, m) * sin(j * psi)
        d1 = - a * sincos[j % 4] * 0.5 * (jv(j-1, m) - jv(j+1, m)) * sin(j * psi)
        d2 = - a * sincos[(j + 1) % 4] * jv(j, m) * sin(j * psi)
        d3 = - a * sincos[j % 4] * jv(j, m) * j * cos(j * psi)

        # Use the pre-calculated model value for I_n
        ydiff = data[i] - model_i[model_idx]

        b_g_mat[0]  += ydiff * d0
        b_g_mat[1]  += ydiff * d1
        b_g_mat[2]  += ydiff * d2
        b_g_mat[3]  += ydiff * d3
        a_g_mat[0]  += d0 * d0 #a11
        a_g_mat[1]  += d0 * d1 #a12
        a_g_mat[2]  += d0 * d2 #a13
        a_g_mat[3]  += d0 * d3 #a14
        a_g_mat[5]  += d1 * d1 #a22
        a_g_mat[6]  += d1 * d2 #a23
        a_g_mat[7]  += d1 * d3 #a24
        a_g_mat[10] += d2 * d2 #a33
        a_g_mat[11] += d2 * d3 #a34
        a_g_mat[15] += d3 * d3 #a44
        
        ssq += ydiff**2

    a_g_mat[4] = a_g_mat[1]
    a_g_mat[8] = a_g_mat[2]
    a_g_mat[9] = a_g_mat[6]
    a_g_mat[12] = a_g_mat[3]
    a_g_mat[13] = a_g_mat[7]
    a_g_mat[14] = a_g_mat[11]

    # ===================================================================
    # START: Diagnostic Print Block 2
    # ===================================================================
    # print(f"\nCalculated SSQ = {ssq:e}")
    # print("="*80)
    # import sys
    # sys.exit("Exiting for debug after first call to coeffs.")
    # ===================================================================
    # END: Diagnostic Print Block 2
    # ===================================================================

    return ssq, a_g_mat, b_g_mat

@profile
def gaussj(a, n, b):
    ipiv = np.zeros(n)
    indxr = np.zeros(NPARMS)
    indxc = np.zeros(NPARMS)

    for i in range(n):
        big = 0.0
        for j in range(n):
            if (ipiv[j] != 1):
                for k in range(n):
                    if (ipiv[k] == 0):
                        if (abs(a[j*n+k]) >= big):
                            big = abs(a[j*n+k])
                            irow = j
                            icol = k 
                    elif (ipiv[k] > 1):
                        print("Gauss-Jordan failed")
                        exit()

        ipiv[icol] += 1
        if (irow != icol):
            for l in range(n):
                a[irow*n+l], a[icol*n+l] = a[icol*n+l], a[irow*n+l]
            b[irow], b[icol] = b[icol], b[irow]
        indxr[i] = irow;
        indxc[i] = icol;
        if (a[icol * n + icol] == 0.0):
            print("Gauss-Jordan failed")
            exit()
        pivinv = 1.0 / a[icol*n+icol]
        a[icol*n+icol] = 1.0
        for l in range(n):
            a[icol*n+l] *= pivinv
        b[icol] *= pivinv
        for ll in range(n):
            if (ll != icol):
                dum = a[ll*n+icol]
                a[ll*n+icol] = 0.0
                for l in range(n):
                    a[ll*n+l] -= a[icol*n+l] * dum
                b[ll] -= b[icol] * dum;

    for l in range(n-1, 0, -1):
        if (indxr[l] != indxc[l]):
            for k in range(n):
                index_r = int(k*n+indxr[l])
                index_c = int(k*n+indxc[l])
                # print(str(index_r))
                # print(str(index_c))
                a[index_r], a[index_c] = a[index_c], a[index_r]

    return a, b

@profile
def ssqf(ndata, data, param):
    bes = np.zeros(MAXDATA + 3)
    sincos = np.zeros(4)

    sincos[1] = -sin(param[2])
    sincos[2] = -cos(param[2])
    sincos[3] = -sincos[1]
    sincos[0] = -sincos[2]

    for i in range(ndata + 3):
        bes[i] = jv(i, param[1])

    c2 = 0.0

    for i in range(ndata):
        j = i+1
        ycalc = param[0] * sincos[j%4] * bes[j] * cos(j*param[3])
        ydiff = data[i] - ycalc
        c2 += ydiff**2
    for i in range(ndata, 2*ndata):
        j = (i - ndata) + 1
        ycalc = param[0] * sincos[j % 4] * bes[j] * (-1)*sin(j * param[3])
        ydiff = data[i] - ycalc
        c2 += ydiff**2

    return c2


@profile
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


@profile
def marq4(ndata, data, parm):
    """
    Performs one robust step of the Levenberg-Marquardt algorithm.
    It tries increasing values of the damping parameter 'lam' until
    it finds a step that improves the sum of squares.
    """
    
    # 1. Calculate the ssq and Jacobian at the current point
    ssq0, a_g_mat, b_g_mat = coeffs(ndata, data, parm)
    
    # 2. Start with a small damping factor (more like Gauss-Newton)
    #    Using a list of explicit values is simple and robust.
    #    You can also generate this list logarithmically.
    lambda_candidates = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    
    best_ssq = ssq0
    best_parm = parm.copy()

    for lam in lambda_candidates:
        # Calculate the proposed parameter step for this lambda
        dp = msolve(lam, a_g_mat, b_g_mat)
        
        # If the step is negligible, no point in continuing
        if np.linalg.norm(dp) < 1e-15:
            continue

        # Calculate the new parameters and the resulting ssq
        p_try = parm + dp
        ssq_try = ssqf(ndata, data, p_try)
        
        # If this lambda gives a better result, save it and we are done for this iteration.
        # The LMA doesn't require finding the *absolute best* lambda, just one
        # that makes progress. This is much faster and more robust than brent.
        if ssq_try < best_ssq:
            best_ssq = ssq_try
            best_parm = p_try
            # We found an improvement, so we can stop searching for a better lambda
            break 
            
    # If no lambda improved the fit, best_ssq will still be ssq0.
    # The 'improve' value will be 0, and the main loop will terminate.
    improve = ssq0 - best_ssq
    
    return best_parm, best_ssq, improve

@profile
def fit(ndata, data, parm):
    nsteps = 0

    while (1):
        parm, fitssq, improve = marq4(ndata, data, parm)
        nsteps += 1

        if (improve < MAR_END):
            status = 1
            break

    if ((status == 1) and (fitssq < FITOK)):
        retssq = fitssq

        if (parm[0] < 0):
            parm[0] = - parm[0]
            parm[2] = parm[2] + np.pi # phasetrack

        if (parm[1] < 0):
            parm[1] = - parm[1]
            parm[2] = parm[2] + np.pi # phasetrack

        return 0, parm, retssq

    bestssq = 9e99
    psitry = 0.0

    for mtry in np.arange(MMIN, MMAX + 1, MSTEP):

        sinsum, cossum = 0.0, 0.0
        nsin, ncos = 0, 0
        for i in range(ndata):
            j = i + 1
            bes = jv (j, mtry) * cos(j*psitry);
            if (abs(bes) > BESLIM):
                if (j%4 == 0):
                    cossum += data[i] / bes
                    ncos += 1
                elif (j%4 == 1):
                    sinsum -= data[i] / bes
                    nsin += 1
                elif (j%4 == 2):
                    cossum -= data[i] / bes
                    ncos += 1
                elif (j%4 == 3):
                    sinsum += data[i] / bes
                    nsin += 1

        for i in range(ndata,2*ndata):
            j = (i - ndata) + 1
            bes = jv (j, mtry) * (-1)*sin(j*psitry)
            if (abs(bes) > BESLIM):
                if (j%4 == 0):
                    cossum += data[i] / bes
                    ncos += 1
                elif (j%4 == 1):
                    sinsum -= data[i] / bes
                    nsin += 1
                elif (j%4 == 2):
                    cossum -= data[i] / bes
                    ncos += 1
                elif (j%4 == 3):
                    sinsum += data[i] / bes
                    nsin += 1

        assert ((nsin > 0) and (ncos > 0))
        ptry = np.arctan2(sinsum / nsin, cossum / ncos)
        asum = 0
        na = 0
        sincos = np.zeros(4)
        sincos[0] = cos (ptry);
        sincos[1] = -sin (ptry);
        sincos[2] = -sincos[0];
        sincos[3] = -sincos[1];

        for i in range(ndata):
            j = i + 1
            bes = jv(j, mtry) * cos(j*psitry)
            if (abs(bes) > BESLIM):
                if (abs(sincos[j%4]) > SINLIM):
                    asum += data[i] / sincos[j%4] / bes
                    na += 1

        for i in range(ndata, 2*ndata):
            j = (i - ndata) + 1
            bes = jv(j, mtry) * (-1) * sin(j*psitry)
            if (abs(bes) > BESLIM):
                if (abs(sincos[j%4]) > SINLIM):
                    asum += data[i] / sincos[j%4] / bes
                    na += 1

        assert (na > 0)
        atry = asum / na
        testp = np.zeros(4)
        testp[0] = atry
        testp[1] = mtry
        testp[2] = ptry
        testp[3] = psitry

        testssq = ssqf(ndata, data, testp)

        if (testssq < bestssq):
            bestssq = testssq
            parm = testp.copy()

    while (1):
        parm, fitssq, improve = marq4(ndata, data, parm)
        if (improve < MAR_END):
            break

    if (parm[0] < 0):
        parm[0] = - parm[0]
        parm[2] = parm[2] + np.pi # phasetrack

    if (parm[1] < 0):
        parm[1] = - parm[1]
        parm[2] = parm[2] + np.pi # phasetrack

    retssq = fitssq

    if (fitssq < FITOK):
        return 1, parm, retssq
    else:
        return 2, parm, retssq




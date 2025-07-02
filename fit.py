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



def calculate_quadratures(n, data, w0, bufferSize):
    Q_data = np.zeros(bufferSize)
    I_data = np.zeros(bufferSize)
    t_step = 0
    for j in range(bufferSize):
        Q_data[j] = data[j]*cos((n+1)*w0*t_step)
        I_data[j] = data[j]*sin((n+1)*w0*t_step)
        t_step += 1

    # t_step = 0
    # Q_data = np.zeros(bufferSize)
    # I_data = np.zeros(bufferSize)
    # for j in range(bufferSize):
    #   np.multiply(data[j], cos((n+1)*w0*t_step), out=Q_data[j])
    #   np.multiply(data[j], sin((n+1)*w0*t_step), out=I_data[j])
        # t_step += 1

    # Q_data = np.multiply(np.array(data),np.cos((n+1)*w0*np.arange(0,bufferSize,1)))
    # I_data = np.multiply(np.array(data),np.sin((n+1)*w0*np.arange(0,bufferSize,1)))

    return Q_data, I_data


def mean_filter(signal):
	return signal.mean()


def coeffs(ndata, data, param):
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

    for i in range(ndata):
        j = i+1

        d0 = sincos[j % 4] * jv(j, m) * cos(j * psi)
        d1 = a * sincos[j % 4] * 0.5 * (jv(j-1, m) - jv(j+1, m)) * cos(j * psi)
        d2 = a * sincos[(j + 1) % 4] * jv(j, m) * cos(j * psi)
        d3 = - a * sincos[j % 4] * jv(j, m) * j * sin(j * psi)

        ydiff = data[i] - a * sincos[j % 4] * jv(j, m) * cos(j * psi)

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
        j = (i - ndata) + 1

        d0 = - sincos[j % 4] * jv(j, m) * sin(j * psi)
        d1 = - a * sincos[j % 4] * 0.5 * (jv(j-1, m) - jv(j+1, m)) * sin(j * psi)
        d2 = - a * sincos[(j + 1) % 4] * jv(j, m) * sin(j * psi)
        d3 = - a * sincos[j % 4] * jv(j, m) * j * cos(j * psi)

        ydiff = data[i] - a * sincos[j % 4] * jv(j, m) * (-1) * sin(j * psi)

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

    return ssq, a_g_mat, b_g_mat



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



def linsolve(a, n, b):
    res = np.zeros(NPARMS)

    for i in range(n):
        rms2 = 0
        for j in range(n):
            rms2 += a[i*n+j]**2
        fac = 1./sqrt(rms2/n)
        for j in range(n):
            a[i*n+j] *= fac
        b[i] *= fac

    ainv = a.copy()
    bsol = b.copy()

    ainv, bsol = gaussj(ainv, n, bsol)

    rms2 = 0

    for i in range(n):
        res[i] = -b[i]
        for j in range(n):
            res[i] += a[i*n+j] * bsol[j]
        rms2 += res[i]**2

    ainv = a.copy()

    ainv, res = gaussj(ainv, n, res)

    for i in range(n):
        bsol[i] -= res[i]

    rms2 = 0

    for i in range(n):
        res[i] = -b[i]
        for j in range(n):
            res[i] += a[i*n+j] * bsol[j]
        rms2 += res[i]**2

    b = bsol.copy();

    return a, b



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



def msolve(lam, a_g_mat, b_g_mat):
    a2 = a_g_mat.copy()

    a2[0]  *= (1+lam)
    a2[5]  *= (1+lam)
    a2[10] *= (1+lam)
    a2[15] *= (1+lam)

    dp = b_g_mat.copy()

    a2, dp = linsolve(a2, 4, dp)

    return dp



def brentfunc(x, ndata, data, p, a_g_mat, b_g_mat):
    px = p + msolve(exp(x), a_g_mat, b_g_mat)
    sqtry = ssqf(ndata, data, px)

    return sqtry, px



def brentfunc2(x, ndata, data, p, a_g_mat, b_g_mat):
    px = p + msolve(exp(x), a_g_mat, b_g_mat)
    sqtry = ssqf(ndata, data, px)

    return sqtry



@with_goto
def LISOfmin(ax, bx, tol, ndata, data, parm, a_g_mat, b_g_mat):
    from numpy import sqrt
    c = 0.381966011250105
    d = 0
    eps = DBL_EPSILON
    tol1 = eps + 1.0
    eps = sqrt(eps)
    a = ax
    b = bx
    v = a + c * (b - a)
    w = v
    x = v
    e = 0.0
    fx, np = brentfunc(x, ndata, data, parm, a_g_mat, b_g_mat)
    fv = fx
    fw = fx
    tol3 = tol / 3.


    label .L20
    xm = (a + b) * .5
    tol1 = eps * abs(x) + tol3
    t2 = tol1 * 2.0
    if (abs(x - xm) <= t2 - (b - a) * .5):
        goto .L190
    p, q, r = 0.0, 0.0, 0.0
    if (abs(e) <= tol1):
        goto .L50
    r = (x - w) * (fx - fv)
    q = (x - v) * (fx - fw)
    p = (x - v) * q - (x - w) * r
    q = (q - r) * 2.0
    if (q <= 0.0):
        goto .L30
    p = -p
    goto .L40


    label .L30
    q = -q


    label .L40
    r = e
    e = d


    label .L50
    if ((abs(p) >= abs(q * .5 * r)) or (p <= q *(a - x)) or (p >= q * (b - x))):
        goto .L60
    d = p / q
    u = x + d
    if ((u - a >= t2) and (b - u >= t2)):
        goto .L90
    d = tol1
    if (x >= xm):
        d = -d
    goto .L90


    label .L60
    if (x >= xm):
        goto .L70
    e = b - x
    goto .L80


    label .L70
    e = a - x


    label .L80
    d = c * e


    label .L90
    if (abs(d) < tol1):
        goto .L100
    u = x + d
    goto .L120


    label .L100
    if (d <= 0.0):
        goto .L110
    u = x + tol1
    goto .L120


    label .L110
    u = x - tol1


    label .L120
    fu, np = brentfunc(u, ndata, data, parm, a_g_mat, b_g_mat)
    if (fx > fu):
        goto .L140
    if (u >= x):
        goto .L130
    a = u
    goto .L140


    label .L130
    b = u


    label .L140
    if (fu > fx):
        goto .L170
    if (u >= x):
        goto .L150
    b = x
    goto .L160


    label .L150
    a = x


    label .L160
    v = w
    fv = fw
    w = x
    fw = fx
    x = u
    fx = fu
    goto .L20


    label .L170
    if ((fu > fw) and (w != x)):
        goto .L180
    v = w
    fv = fw
    w = u
    fw = fu


    label .L180
    if ((fu > fv) and (v != x) and (v != w)):
        goto .L20
    v = u
    fv = fu
    goto .L20


    label .L190
    return x



def marq4(ndata, data, parm):

    ssq0, a_g_mat, b_g_mat = coeffs(ndata, data, parm)
    lmin = LISOfmin (log(LAMBDA_MIN), log(LAMBDA_MAX), BRENT_TOL, ndata, data, parm, a_g_mat, b_g_mat)
    ssq1, newparm = brentfunc(lmin, ndata, data, parm, a_g_mat, b_g_mat)

    if (ssq1 < ssq0):
        parm = newparm.copy()
        ssq = ssq1
    else:
        ssq = ssq0

    return parm, ssq, ssq0 - ssq1


def marq4_v2(ndata, data, parm):

    ssq0, a_g_mat, b_g_mat = coeffs(ndata, data, parm)
    lmin = brent(brentfunc2, args=(ndata, data, parm, a_g_mat, b_g_mat), tol=BRENT_TOL)
    ssq1, newparm = brentfunc(lmin, ndata, data, parm, a_g_mat, b_g_mat)

    if (ssq1 < ssq0):
        parm = newparm.copy()
        ssq = ssq1
    else:
        ssq = ssq0

    return parm, ssq, ssq0 - ssq1


def fit(ndata, data, parm):
    nsteps = 0

    while (1):
        parm, fitssq, improve = marq4_v2(ndata, data, parm)
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
        parm, fitssq, improve = marq4_v2(ndata, data, parm)
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




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
import scipy as sp
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

def generate_test_signal(
    fs: float,
    fm: float,
    duration: float,
    m1: float = 5.0,
    m2: float = 0.5,
    im_depth: float = 0.1,
    im_freq: float = 0.1,
):
    """
    Toy signal for sanity-checking the API
    """
    t = np.arange(duration*fs)/fs
    omega = 2 * np.pi * fm

    phi_mod = m1 * np.sin(omega * t) + m2 * np.sin(2*omega * t)

    v_meas = np.cos(phi_mod)

    return t, phi_mod, v_meas

def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    # Safety check for filter stability
    if normal_cutoff >= 1.0:
        return data
        
    b, a = sp.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = sp.signal.filtfilt(b, a, data)
    return y

def correct_intensity_modulation(
    v_meas: np.ndarray, 
    fs: float, 
    fm: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Intensity modulation correction.
    
    Removes the additive (I0) and multiplicative (Iv) intensity modulations
    that often accompany laser current modulation.

    Parameters
    ----------
    v_meas : np.ndarray
        Raw photodetector voltage signal.
    fs : float
        Sampling frequency (Hz).
    fm : float
        Modulation frequency (Hz).

    Returns
    -------
    v_corrected : np.ndarray
        The normalized interferometric signal (approx. cos(phi)).
    I0 : np.ndarray
        The estimated additive intensity term.
    Iv : np.ndarray
        The estimated multiplicative envelope term.
    """
    raise NotImplementedError

def hilbert_transform(fx):
    """
    Approximate the Hilbert transform of a real-valued function f(x)
    sampled on a uniform grid, using only NumPy.

    Parameters
    ----------
    fx : array_like
        1D array of real samples of f(x) on an equally spaced grid.

    Returns
    -------
    Hf : ndarray
        1D array with the Hilbert transform H[f](x) evaluated at the
        same grid points.
    """
    fx = np.asarray(fx, dtype=float)
    N = fx.shape[0]

    # Fourier transform of the real signal
    F = np.fft.fft(fx)

    # Construct the multiplier that creates the analytic signal
    # (this is equivalent to multiplying by -i*sgn(ω) in the continuous case)
    h = np.zeros(N, dtype=float)
    if N % 2 == 0:
        # even length
        h[0] = 1.0
        h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        # odd length
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0

    # Apply the filter and inverse FFT to get the analytic signal
    analytic = np.fft.ifft(F * h)

    # Imaginary part is the Hilbert transform
    return np.imag(analytic)

def hilbert_pv_integral(fx, x):
    """
    Approximate the Hilbert transform using the principal-value
    integral on a finite, uniformly spaced grid.

    Hf(x_j) ≈ (Δx/π) sum_{k ≠ j} f(x_k) / (x_j - x_k)

    Parameters
    ----------
    fx : array_like
        Real-valued samples f(x_k).
    x : array_like
        Sample locations x_k, assumed 1D and uniformly spaced.

    Returns
    -------
    Hf : ndarray
        Approximate Hilbert transform Hf(x_j) at the same grid points.
    """
    fx = np.asarray(fx, dtype=float)
    x = np.asarray(x, dtype=float)

    if fx.shape != x.shape:
        raise ValueError("fx and x must have the same shape")

    # Check uniform spacing (not strictly necessary but safer)
    dx = np.diff(x)
    if not np.allclose(dx, dx[0]):
        raise ValueError("x must be uniformly spaced")

    dx = dx[0]
    N = x.size

    # Build matrix of differences x_j - x_k
    X = x[:, None]      # shape (N,1)
    Y = x[None, :]      # shape (1,N)
    denom = X - Y       # shape (N,N)

    # Implement principal value: ignore k=j term by making it infinite
    np.fill_diagonal(denom, np.inf)

    # Compute PV sum: (dx/pi) * sum_k f(x_k)/(x_j - x_k)
    Hf = (dx / np.pi) * np.sum(fx[None, :] / denom, axis=1)

    return Hf

def identify_local_minima(phi: np.ndarray) -> np.ndarray:
    """
    """
    return sp.signal.argrelextrema(phi, np.less)[0]

def stitch_segments(phi: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, slice]:
    """
    Implement

      Φ_reconstr(t_k) =
        Φ_H(t_k) - Φ_H(t_{q0})                          for q0 ≤ k < q1
        Φ_H(t_k) - Φ_H(t_{qj}) + Φ_reconstr(t_{qj-1})   for qj ≤ k < q_{j+1}, j even
       -Φ_H(t_k) + Φ_H(t_{qj}) + Φ_reconstr(t_{qj-1})   for qj ≤ k < q_{j+1}, j odd

    where phi[k] = Φ_H(t_k) and q[j] = q_j.
    """

    phi = np.asarray(phi)
    q = np.asarray(q, dtype=int)

    if q.size < 2:
        raise ValueError("q must contain at least q0 and q1")

    phi_rec = np.zeros_like(phi, dtype=float)

    # First interval: q0 ≤ k < q1
    q0, q1 = q[0], q[1]
    phi_rec[q0:q1] = phi[q0:q1] - phi[q0]

    # Remaining intervals: qj ≤ k < q_{j+1}, j = 1, 2, ...
    for j in range(1, len(q) - 1):
        start, stop = q[j], q[j + 1]

        if j % 2 == 0:  # j even
            phi_rec[start:stop] = (
                phi[start:stop] - phi[start] + phi_rec[start - 1]
            )
        else:           # j odd
            phi_rec[start:stop] = (
                -phi[start:stop] + phi[start] + phi_rec[start - 1]
            )

    return phi_rec

def reconstruct_waveform(
    v_meas: np.ndarray,
    fs: float,
    fm: float,
    correct_intensity: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Main API function for Shmagun Waveform Reconstruction.

    Parameters
    ----------
    v_meas : np.ndarray
        Input voltage array from witness interferometer.
    fs : float
        Sampling rate (Hz).
    fm : float
        Nominal modulation frequency (Hz).
    correct_intensity : bool
        Whether to apply IM correction. Defaults to False.

    Returns
    -------
    phi_rec : np.ndarray
        The reconstructed effective phase modulation waveform.
    q : list of int
        Indices of stitch locations.
    """
    # Pre-processing
    if correct_intensity:
        v_proc, _, _ = correct_intensity_modulation(v_meas, fs, fm)
    else:
        v_proc = v_meas.copy()

    # Hilbert Transform + Phase Extraction
    phi_hilbert_wrapped = np.angle(sp.signal.hilbert(v_proc))
    phi_hilbert = np.unwrap(phi_hilbert_wrapped, period=np.pi)

    # Calculate symmetric discrete first derivative of the phase signal
    dphi = 0.5 * (np.roll(phi_hilbert, -1) - np.roll(phi_hilbert, 1))
    dphi[0] = dphi[-1] = 0

    # Low-pass filter of phase derivative
    dphi_filtered = butter_lowpass_filter(
        data=dphi, cutoff=5*fm, fs=fs, order=2
    )

    # Turning points
    q = identify_local_minima(dphi_filtered)

    if len(q) < 2:
        print("Warning: insufficient turning points.")
        exit()

    # Segment stitching
    phi_rec_full = stitch_segments(phi_hilbert, q)

    # Return full reconstruction and region indices
    return phi_rec_full, phi_hilbert_wrapped, phi_hilbert, dphi, dphi_filtered, q


if __name__ == "__main__":
    fs = 200e4         # sampling rate
    fm = 1e3           # modulation frequency
    duration = 5/fm   # 50 periods

    t, phi_mod, v_meas = generate_test_signal(fs, fm, duration, m1=50.0, m2=0.0, im_depth=0.0, im_freq=0.0)

    # Run Shmagun-style reconstruction
    phi_rec_full, phi_hilbert_wrapped, phi_hilbert, dphi, dphi_filtered, q = reconstruct_waveform(v_meas, fs, fm, correct_intensity=False)

    q0 = q[0]
    qlast = q[-1]

    # Compare on matching central slice
    t_slice = t[q0:qlast]
    phi_rec_slice = phi_rec_full[q0:qlast]
    phi_true_slice = phi_mod[q0:qlast]
    n_rec = len(phi_true_slice)

    phi_rec0 = phi_rec_slice - np.mean(phi_rec_slice)
    phi_true0 = phi_true_slice - np.mean(phi_true_slice)

    # account for possible global sign ambiguity
    err_plus = np.linalg.norm(phi_rec0 - phi_true0) / np.sqrt(n_rec)
    err_minus = np.linalg.norm(phi_rec0 + phi_true0) / np.sqrt(n_rec)
    if err_minus < err_plus:
        sign = -1.0
        err = err_minus
    else:
        sign = 1.0
        err = err_plus

    corr = np.corrcoef(sign * phi_rec0, phi_true0)[0, 1]

    print(f"Reconstruction length: {n_rec} samples")
    print(f"RMS error (best sign): {err:.3f} rad")
    print(f"Correlation (best sign): {corr:.8f}")
    print("First 10 samples of phi_true_slice:")
    print(phi_true_slice[:10])
    print("First 10 samples of phi_rec (up to sign/offset):")
    print(phi_rec_slice[:10])

    fig, ax = plt.subplots(figsize=(20,3), dpi=150)
    ax.set_title('Signal')
    ax.plot(t, v_meas)
    ax.plot(t, np.real(sp.signal.hilbert(v_meas)), ls='--')
    # plt.show()

    fig, ax = plt.subplots(figsize=(20,3), dpi=150)
    ax.set_title('Hilbert transform of signal')
    ax.plot(t, np.imag(sp.signal.hilbert(v_meas)))
    ax.plot(t, hilbert_transform(v_meas), ls='--')
    # plt.show()


    fig, ax = plt.subplots(figsize=(20,3), dpi=150)
    ax.set_title('Phase extraction and derivative')
    ax.plot(t, phi_hilbert_wrapped)
    ax.plot(t, phi_hilbert)
    ax.plot(t, dphi)
    ax.plot(t, dphi_filtered)
    for idx in q:
        ax.axvline(t[idx], c='k', ls='--')
    # plt.show()

    fig, ax = plt.subplots(figsize=(20,3), dpi=150)
    ax.set_title('Wavefront reconstruction')
    ax.plot(t_slice, phi_true0)
    ax.plot(t_slice, -phi_rec0, ls='--')
    plt.show()



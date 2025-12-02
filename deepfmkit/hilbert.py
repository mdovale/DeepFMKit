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
):
    """
    Generate a synthetic interferometric signal for testing reconstruction algorithms.

    Simulates a signal of the form:
    V(t) = cos(phi(t))
    where phi(t) contains harmonic modulation.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    fm : float
        Fundamental modulation frequency in Hz.
    duration : float
        Signal duration in seconds.
    m1 : float, optional
        Modulation index for the fundamental frequency. Default is 5.0.
    m2 : float, optional
        Modulation index for the second harmonic. Default is 0.5.

    Returns
    -------
    t : np.ndarray
        Time vector.
    phi_mod : np.ndarray
        The true phase modulation waveform.
    v_meas : np.ndarray
        The simulated voltage signal with intensity modulation applied.
    """
    t = np.arange(duration*fs)/fs
    omega = 2 * np.pi * fm

    phi_mod = m1 * np.sin(omega * t) + m2 * np.sin(2*omega * t)

    v_meas = np.cos(phi_mod)

    return t, phi_mod, v_meas

def butter_lowpass_filter(
    data: np.ndarray, 
    cutoff: float, 
    fs: float, 
    order: int = 2
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth low-pass filter to the input data.

    Parameters
    ----------
    data : np.ndarray
        Input signal (1D).
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Order of the filter. Default is 2.

    Returns
    -------
    y : np.ndarray
        Filtered signal.

    Raises
    ------
    ValueError
        If cutoff frequency is greater than or equal to the Nyquist frequency.
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
    TODO Implement correct_intensity_modulation
    """
    raise NotImplementedError

def hilbert_transform(fx: np.ndarray) -> np.ndarray:
    """
    Compute the Hilbert transform of a real-valued signal using FFT.

    H[f](t) = Im( Analytic_Signal(f(t)) )

    Parameters
    ----------
    fx : np.ndarray
        1D array of real-valued samples.

    Returns
    -------
    Hf : np.ndarray
        The Hilbert transform of the input.
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

def hilbert_pv_integral(fx: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Approximate Hilbert transform via Principal Value (PV) integral on a grid.
    
    Warning: This implementation is O(N^2) and is significantly slower 
    than FFT-based methods. Intended for educational verification.

    Parameters
    ----------
    fx : np.ndarray
        Real-valued samples f(x).
    x : np.ndarray
        Uniformly spaced grid points.

    Returns
    -------
    Hf : np.ndarray
        Approximate Hilbert transform.
    
    Raises
    ------
    ValueError
        If inputs shape mismatch or x is not uniformly spaced.
    """
    fx = np.asarray(fx, dtype=float)
    x = np.asarray(x, dtype=float)

    if fx.shape != x.shape:
        raise ValueError(f"Shape mismatch: fx {fx.shape}, x {x.shape}")

    dx_arr = np.diff(x)
    if not np.allclose(dx_arr, dx_arr[0], rtol=1e-5):
        raise ValueError("Grid x must be uniformly spaced.")

    dx = dx_arr[0]
    
    # Broadcasting to create difference matrix (x_j - x_k)
    # Memory Warning: creates an N x N array.
    denom = x[:, None] - x[None, :]
    
    # PV: Exclude diagonal (j=k) by treating as infinity (result -> 0)
    np.fill_diagonal(denom, np.inf)

    # Discrete Hilbert Transform Summation
    Hf = (dx / np.pi) * np.sum(fx[None, :] / denom, axis=1)

    return Hf

def identify_local_minima(data: np.ndarray) -> np.ndarray:
    """
    Find indices of local minima in a 1D array.

    Parameters
    ----------
    data : np.ndarray
        Input 1D array.

    Returns
    -------
    indices : np.ndarray
        Array of indices where local minima occur.
    """
    return sp.signal.argrelextrema(data, np.less)[0]

def stitch_segments(phi: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full phase waveform by stitching Hilbert segments.
    
    This function implements the "Shmagun" stitching logic where phase segments
    between turning points (q) are flipped and offset to recover directionality of the
    reconstructed effective modulation wavefront.

    Parameters
    ----------
    phi : np.ndarray
        The unwrapped Hilbert phase (usually from an analytic signal).
    q : np.ndarray
        Indices of the turning points (local minima of phase derivative).

    Returns
    -------
    phi_rec : np.ndarray
        The reconstructed continuous waveform.
        
    Raises
    ------
    ValueError
        If fewer than 2 turning points are provided.
    """
    phi = np.asarray(phi)
    q = np.asarray(q, dtype=int)

    if q.size < 2:
        raise ValueError("At least 2 turning points (q) are required for stitching.")

    phi_rec = np.zeros_like(phi, dtype=float)

    # 1. Initialize first interval: q0 <= k < q1
    q0, q1 = q[0], q[1]
    phi_rec[q0:q1] = phi[q0:q1] - phi[q0]

    # 2. Iterative stitching
    # Logic: Align the start of the current segment with the end of the previous one.
    # The sign flips every segment (even/odd) to counteract fold-over.
    for j in range(1, len(q) - 1):
        start, stop = q[j], q[j + 1]
        prev_val = phi_rec[start - 1]
        
        curr_segment = phi[start:stop]
        anchor = phi[start]

        if j % 2 == 0:
            # Even segment: preserve slope direction
            phi_rec[start:stop] = (curr_segment - anchor) + prev_val
        else:
            # Odd segment: invert slope direction
            phi_rec[start:stop] = (-curr_segment + anchor) + prev_val

    return phi_rec

def reconstruct_waveform(
    v_meas: np.ndarray,
    fs: float,
    fm: float,
    correct_intensity: bool = False,
    lp_cutoff_factor: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct phase waveform from interferometric fringe signal.

    Algorithm Pipeline:
    1. (Optional) Intensity correction.
    2. Hilbert transform to extract wrapped phase.
    3. Differentiation of phase.
    4. Identification of turning points (where d_phase ~ 0).
    5. Segment stitching to unwrap the high-modulation waveform.

    Parameters
    ----------
    v_meas : np.ndarray
        Measured voltage signal.
    fs : float
        Sampling rate (Hz).
    fm : float
        Nominal modulation frequency (Hz).
    correct_intensity : bool, optional
        If True, applies AM/DC correction before processing. Default False.
    lp_cutoff_factor : float, optional
        Multiplier for fm to determine low-pass cutoff for the phase derivative.
        Default is 5.0 (cutoff = 5 * fm).

    Returns
    -------
    phi_rec_full : np.ndarray
        Reconstructed phase (zeros outside the stitching range).
    phi_hilbert_wrapped : np.ndarray
        Raw phase from Hilbert transform [-pi, pi].
    phi_hilbert : np.ndarray
        Standard unwrapped Hilbert phase.
    dphi : np.ndarray
        Derivative of the Hilbert phase.
    dphi_filtered : np.ndarray
        Filtered derivative used for turning point detection.
    q : np.ndarray
        Indices of detected turning points.

    Raises
    ------
    RuntimeError
        If insufficient turning points are found to reconstruct the signal.
    """
    # 1. Pre-processing
    if correct_intensity:
        v_proc, _, _ = correct_intensity_modulation(v_meas, fs, fm)
    else:
        v_proc = v_meas.copy()

    # 2. Hilbert Transform + Phase Extraction
    analytic_sig = sp.signal.hilbert(v_proc)
    phi_hilbert_wrapped = np.angle(analytic_sig)
    phi_hilbert = np.unwrap(phi_hilbert_wrapped, period=np.pi)

    # 3. Calculate symmetric discrete first derivative
    # (Central difference)
    dphi = np.zeros_like(phi_hilbert)
    dphi[1:-1] = 0.5 * (phi_hilbert[2:] - phi_hilbert[:-2])
    # Boundary handling (replicate neighbors)
    dphi[0] = dphi[1]
    dphi[-1] = dphi[-2]

    # 4. Low-pass filter the derivative to find robust minima
    dphi_filtered = butter_lowpass_filter(
        data=dphi, 
        cutoff=lp_cutoff_factor * fm, 
        fs=fs, 
        order=2
    )

    # 5. Identify Turning Points (Local Minima of Derivative)
    q = identify_local_minima(dphi_filtered)

    if len(q) < 2:
        raise RuntimeError(
            "Insufficient turning points detected. Check signal quality, "
            "modulation index, or filter cutoff settings."
        )

    # 6. Stitch Segments
    phi_rec_full = stitch_segments(phi_hilbert, q)

    return phi_rec_full, phi_hilbert_wrapped, phi_hilbert, dphi, dphi_filtered, q


if __name__ == "__main__":
    # --- Integration Test / Demo ---
    
    # Simulation Parameters
    FS = 200e4          # 2 MHz sampling
    FM = 1e3            # 1 kHz modulation
    DURATION = 5.0 / FM # 5 periods
    
    # Generate Signal with Intensity Modulation artifacts
    t, phi_true, v_meas = generate_test_signal(
        FS, FM, DURATION, 
        m1=50.0, m2=0.0, 
    )

    print("Processing signal...")
    try:
        # Run Reconstruction (with IM correction enabled)
        phi_rec_full, _, _, dphi, dphi_filt, q = reconstruct_waveform(
            v_meas, FS, FM, correct_intensity=False
        )

        # Evaluate performance on the valid stitched region
        q_start, q_end = q[0], q[-1]
        
        # Slicing
        phi_rec_slice = phi_rec_full[q_start:q_end]
        phi_true_slice = phi_true[q_start:q_end]
        
        # Remove DC offsets for comparison
        phi_rec_centered = phi_rec_slice - np.mean(phi_rec_slice)
        phi_true_centered = phi_true_slice - np.mean(phi_true_slice)
        
        # Determine global sign ambiguity (common in interferometry)
        corr_pos = np.corrcoef(phi_rec_centered, phi_true_centered)[0, 1]
        sign = np.sign(corr_pos)
        
        phi_final = sign * phi_rec_centered
        rmse = np.sqrt(np.mean((phi_final - phi_true_centered)**2))

        print(f"Reconstruction Interval: Samples {q_start} to {q_end}")
        print(f"Correlation: {abs(corr_pos):.5f}")
        print(f"RMSE: {rmse:.4f} rad")

        # Visualization
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        axes[0].set_title("Input Signal (with IM)")
        axes[0].plot(t, v_meas, label='Measured V')
        axes[0].legend()
        
        axes[1].set_title("Phase Derivative & Turning Points")
        axes[1].plot(t, dphi, color='gray', alpha=0.5, label='Raw dPhi')
        axes[1].plot(t, dphi_filt, color='blue', label='Filtered dPhi')
        axes[1].plot(t[q], dphi_filt[q], 'ro', label='Turning Points')
        axes[1].legend()

        axes[2].set_title("Reconstructed vs True Waveform")
        axes[2].plot(t[q_start:q_end], phi_true_centered, 'k', label='True')
        axes[2].plot(t[q_start:q_end], phi_final, 'r--', label='Reconstructed')
        axes[2].legend()

        axes[3].set_title("Error Residual")
        axes[3].plot(t[q_start:q_end], phi_final - phi_true_centered, color='orange')
        axes[3].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Reconstruction failed: {e}")
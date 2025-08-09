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
from scipy.signal import sawtooth, square
from typing import Dict, Optional


def cosine(t_phase: np.ndarray) -> np.ndarray:
    """Generates a standard cosine waveform.

    This is the ideal, default waveform for DFMI.

    Parameters
    ----------
    t_phase : np.ndarray
        The phase axis for the waveform, typically `omega_mod * t + psi`.

    Returns
    -------
    np.ndarray
        The resulting unitless cosine waveform.
    """
    return np.cos(t_phase)


def sine(t_phase: np.ndarray) -> np.ndarray:
    """Generates a standard sine waveform.

    Parameters
    ----------
    t_phase : np.ndarray
        The phase axis for the waveform, typically `omega_mod * t + psi`.

    Returns
    -------
    np.ndarray
        The resulting unitless sine waveform.
    """
    return np.sin(t_phase)


def shd(
    t_phase: np.ndarray, distortion_amp: float = 0.0, distortion_phase: float = 0.0
) -> np.ndarray:
    """Generates a waveform with second harmonic distortion (SHD).

    This function models a common non-ideality in signal generators where
    a spurious second harmonic is added to the fundamental tone.

    Parameters
    ----------
    t_phase : np.ndarray
        The phase axis for the fundamental tone, typically `omega_mod * t + psi`.
    distortion_amp : float, optional
        The fractional amplitude of the second harmonic relative to the
        fundamental (which has an amplitude of 1). Defaults to 0.0.
    distortion_phase : float, optional
        The phase of the second harmonic in radians, relative to the phase
        of the fundamental's second harmonic (i.e., relative to `2*t_phase`).
        Defaults to 0.0.

    Returns
    -------
    np.ndarray
        The resulting unitless waveform: `cos(t_phase) + distortion_amp * cos(2*t_phase + distortion_phase)`.
    """
    fundamental = np.cos(t_phase)
    second_harmonic = distortion_amp * np.cos(2 * t_phase + distortion_phase)
    return fundamental + second_harmonic


def triangle_wave(t_phase: np.ndarray, width: float = 0.5) -> np.ndarray:
    """Generates a triangle wave.

    This function uses `scipy.signal.sawtooth` with a `width` of 0.5 to
    produce a standard triangle wave that ramps from -1 to 1 and back to -1
    over a 2*pi period.

    Parameters
    ----------
    t_phase : np.ndarray
        The phase axis for the waveform.
    width : float, optional
        The point in the cycle where the ramp peaks. A value of 0.5 creates
        a symmetric triangle wave. Must be in the interval [0, 1].
        Defaults to 0.5.

    Returns
    -------
    np.ndarray
        The resulting unitless triangle waveform.
    """
    return sawtooth(t_phase, width=width)


def square_wave(t_phase: np.ndarray, duty: float = 0.5) -> np.ndarray:
    """Generates a square wave.

    Parameters
    ----------
    t_phase : np.ndarray
        The phase axis for the waveform.
    duty : float, optional
        The duty cycle of the square wave, representing the fraction of the
        period for which the signal is positive. Must be in the interval
        [0, 1]. Defaults to 0.5 for a standard square wave.

    Returns
    -------
    np.ndarray
        The resulting unitless square waveform, oscillating between -1 and 1.
    """
    return square(t_phase, duty=duty)


def dfm_like_wave(
    t_phase: np.ndarray, harmonics: Optional[Dict[int, float]] = None
) -> np.ndarray:
    """Generates a custom multi-harmonic signal based on a fundamental cosine.

    This allows the creation of arbitrary periodic waveforms by specifying the
    amplitudes of higher harmonics relative to the fundamental.

    Parameters
    ----------
    t_phase : np.ndarray
        The phase axis for the fundamental tone.
    harmonics : dict[int, float], optional
        A dictionary mapping harmonic numbers (int > 1) to their fractional
        amplitudes. For example, `{2: 0.1, 3: 0.05}` adds a 10% 2nd harmonic
        and a 5% 3rd harmonic. If None, a default example is used.
        Defaults to None.

    Returns
    -------
    np.ndarray
        The resulting composite, unitless waveform.
    """
    if harmonics is None:
        # Provide a default example if none is given
        harmonics = {2: 0.1, 3: 0.05}

    # Start with the fundamental tone
    y = np.cos(t_phase)

    # Add higher harmonics
    for n, amp in harmonics.items():
        if n > 1:  # Ensure only higher harmonics are added
            y += amp * np.cos(n * t_phase)

    return y


def dfm_wave(t_phase: np.ndarray, m: float = 1.0, phi: float = 0.0) -> np.ndarray:
    """Generates a waveform shaped like the AC component of an ideal DFMI signal.

    This function calculates `cos(phi + m * cos(t_phase))` and normalizes it.
    It can be used to create a complex modulation waveform whose shape is
    determined by the parameters of a virtual "inner" DFMI system.

    Parameters
    ----------
    t_phase : np.ndarray
        The modulation phase axis, representing `omega_mod * t + psi`.
    m : float, optional
        The effective modulation index of the virtual inner DFMI system,
        in radians. This controls the richness of the harmonic content.
        Defaults to 1.0.
    phi : float, optional
        The interferometric phase of the virtual inner DFMI system, in radians.
        This controls the symmetry and shape of the waveform. Defaults to 0.0.

    Returns
    -------
    np.ndarray
        The resulting unitless waveform, with its DC component removed and
        normalized to have a peak amplitude of approximately 1.
    """
    waveform = np.cos(phi + m * np.cos(t_phase))
    # Remove DC component and normalize to have a peak of ~1
    ac_waveform = waveform - np.mean(waveform)
    peak = np.max(np.abs(ac_waveform))
    if peak > 1e-9:
        return ac_waveform / peak
    return ac_waveform

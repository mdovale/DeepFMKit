# This is the complete content for the new waveforms.py file.
# I have included your existing functions and added the new dfm_wave function.

import numpy as np
from scipy.signal import sawtooth, square

def second_harmonic_distortion(t_phase, distortion_amp=0.0, distortion_phase=0.0):
    """
    Generates a waveform with a fundamental and a second harmonic.

    Parameters
    ----------
    t_phase : np.ndarray
        The phase axis for the fundamental tone (omega_mod * t + psi).
    distortion_amp : float, optional
        The fractional amplitude of the second harmonic.
    distortion_phase : float, optional
        The phase of the second harmonic relative to the fundamental.

    Returns
    -------
    np.ndarray
        The resulting unitless waveform.
    """
    fundamental = np.cos(t_phase)
    second_harmonic = distortion_amp * np.cos(2 * t_phase + distortion_phase)
    return fundamental + second_harmonic

def triangle_wave(t_phase, width=0.5):
    """
    Generates a triangle wave with phase `t_phase`.
    This uses the sawtooth function from scipy.signal, where a width of 0.5
    produces a triangle wave.
    """
    return sawtooth(t_phase, width=width)

def square_wave(t_phase, duty=0.5):
    """
    Generates a square wave with phase `t_phase`.
    
    Parameters
    ----------
    duty : float, optional
        The duty cycle of the square wave, from 0 to 1. Defaults to 0.5.
    """
    return square(t_phase, duty=duty)

def dfm_like_wave(t_phase, harmonics=None):
    """
    Generates a custom multi-harmonic signal based on a fundamental cosine.
    
    Parameters
    ----------
    harmonics : dict, optional
        A dictionary mapping harmonic numbers to their fractional amplitudes.
        Example: {2: 0.1, 3: 0.05} for 10% 2nd and 5% 3rd harmonic.
        Defaults to a simple example if None.
    """
    if harmonics is None:
        harmonics = {2: 0.1, 3: 0.05}
    
    # Fundamental tone
    y = np.cos(t_phase)
    # Add higher harmonics
    for n, amp in harmonics.items():
        y += amp * np.cos(n * t_phase)
    return y

def dfm_wave(t_phase, m=1.0, phi=0.0):
    """
    Generates a waveform shaped like the AC component of an ideal DFMI signal.

    This function calculates `cos(phi + m * cos(t_phase))`. It can be used
    to create a complex modulation waveform whose shape is determined by the
    parameters of an "inner" DFMI system.

    Parameters
    ----------
    t_phase : np.ndarray
        The modulation phase axis, representing `omega_mod * t + psi`.
    m : float, optional
        The effective modulation index of the inner DFMI system, in radians.
        This controls the richness of the harmonic content. Defaults to 1.0.
    phi : float, optional
        The interferometric phase of the inner DFMI system, in radians.
        This controls the symmetry of the waveform. Defaults to 0.0.

    Returns
    -------
    np.ndarray
        The resulting unitless waveform.
    """
    return np.cos(phi + m * np.cos(t_phase))
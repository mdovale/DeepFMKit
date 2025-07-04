import numpy as np

def vectorized_downsample(signal, R):
    """
    Downsamples a 1D signal by averaging over blocks of size R.

    This method uses an efficient, vectorized NumPy approach to perform block
    averaging, also known as boxcar averaging or binning. It reshapes the
    input signal into a 2D array and then computes the mean along the new
    axis, effectively reducing the sampling rate by the factor R.

    If the length of the input signal is not a multiple of the downsampling
    factor R, the signal is trimmed from the end to the largest possible
    length that is a multiple of R. Any remaining samples are discarded.

    Parameters
    ----------
    signal : numpy.ndarray
        The 1D signal array to be downsampled.
    R : int
        The downsampling factor, i.e., the number of samples in each
        averaging block. Must be a positive integer.

    Returns
    -------
    numpy.ndarray
        The downsampled signal. Returns an empty array if R is not a
        positive integer or if the signal is shorter than R.
    """
    # --- 1. Input Validation and Edge Case Handling ---
    # Ensure R is a positive integer. If not, downsampling is not possible.
    if not isinstance(R, int) or R <= 0:
        print(f"Downsampling factor R must be a positive integer, but got {R}. Returning empty array.")
        return np.array([])

    # Ensure the signal is a NumPy array for vectorized operations.
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    # --- 2. Trim the signal to a length that is a multiple of R ---
    # This is necessary for the reshape operation to work correctly.
    trimmed_len = (len(signal) // R) * R

    # If the trimmed length is zero (e.g., signal is shorter than R),
    # there is nothing to downsample.
    if trimmed_len == 0:
        return np.array([])

    # Create a view of the signal with the trimmed length.
    trimmed_signal = signal[:trimmed_len]

    # --- 3. Reshape and compute the mean for downsampling ---
    # Reshape the 1D array into a 2D array of shape (-1, R).
    # The '-1' automatically calculates the correct number of rows.
    # Then, compute the mean along axis=1 to average each block of R samples.
    return trimmed_signal.reshape(-1, R).mean(axis=1)
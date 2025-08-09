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
import pandas as pd
import pytest
from deepfmkit.core import DeepFrame
from deepfmkit.dsp import vectorized_downsample
from deepfmkit.fitters import _calculate_fit_params

# Mark all tests in this file as belonging to the 'fitters' module
pytestmark = pytest.mark.fitters


# A helper function to compare angles, accounting for wrapping at +/- pi
def compare_angles(a1, a2, atol):
    diff = np.abs(a1 - a2)
    # Ensure the shortest distance between angles is within tolerance
    assert np.min([diff, 2 * np.pi - diff]) < atol


# --- Tests for StandardNLSFitter ---


def test_nls_fitter_sequential_vs_parallel(basic_sim_config):
    """
    Validates that the StandardNLSFitter produces identical results when run
    in sequential mode versus parallel mode. This is a critical consistency check.
    """
    dff = DeepFrame()
    dff.add_sim(basic_sim_config)
    dff.simulate(label="test_sim", n_seconds=0.5, mode="snr", snr_db=1000)

    # --- Run in Sequential Mode ---
    fit_seq = dff.fit(
        main_label="test_sim", method="nls", fit_label="fit_sequential", parallel=False
    )
    results_seq = dff.fits_df["fit_sequential"]

    # --- Run in Parallel Mode ---
    fit_par = dff.fit(
        main_label="test_sim",
        method="nls",
        fit_label="fit_parallel",
        parallel=True,
        n_cores=2,  # Use a small number of cores for the test
    )
    results_par = dff.fits_df["fit_parallel"]

    # --- Assertions ---
    assert fit_seq is not None and fit_par is not None
    # Use pandas testing utility to compare the entire DataFrames
    pd.testing.assert_frame_equal(results_seq, results_par, atol=1e-7)


def test_nls_fitter_constrained_fit(basic_sim_config):
    """
    Tests that the high-level NLSFitter correctly handles a constrained fit,
    holding specified parameters constant.
    """
    dff = DeepFrame()

    # Modify the config so the fixed params match the fitter's defaults
    basic_sim_config.laser.psi = 0.0
    basic_sim_config.laser.set_df_for_m(basic_sim_config.ifo, m=6.0)

    dff.add_sim(basic_sim_config)
    dff.simulate(label="test_sim", n_seconds=0.1, mode="snr", snr_db=1000)

    fit_obj = dff.fit(
        main_label="test_sim",
        method="nls",
        fit_params=["amp", "phi"],  # Only fit for amp and phi
        parallel=False,
    )
    results = dff.fits_df[fit_obj.label]

    # Check that the fitted parameters converged to the truth
    assert results["amp"].mean() == pytest.approx(basic_sim_config.laser.amp, rel=1e-6)
    compare_angles(results["phi"].mean(), basic_sim_config.ifo.phi, atol=1e-6)

    # Check that the fixed parameters remained at their initial default values
    assert results["m"].mean() == pytest.approx(6.0)
    assert results["psi"].mean() == pytest.approx(0.0)


def test_nls_fitter_psi_init(basic_sim_config):
    """
    Tests that the smart `psi_init` routine helps convergence when given a
    very bad initial guess for psi.
    """
    dff = DeepFrame()

    # The true psi is -pi/8. We will provide a guess that is almost pi radians away.
    bad_psi_guess = basic_sim_config.laser.psi + np.pi * 0.9

    dff.add_sim(basic_sim_config)
    dff.simulate(label="test_sim", n_seconds=0.1, mode="snr", snr_db=1000)

    # --- Fit WITHOUT smart init (expected to be less accurate or fail) ---
    fit_bad_guess = dff.fit(
        main_label="test_sim",
        method="nls",
        fit_label="fit_bad",
        init_psi=bad_psi_guess,
        init_psi_method=None,
        parallel=False,
    )
    # FIX: Use the correct variable name `fit_bad_guess`
    results_bad = dff.fits_df[fit_bad_guess.label]

    # --- Fit WITH smart init (expected to succeed) ---
    fit_smart_guess = dff.fit(
        main_label="test_sim",
        method="nls",
        fit_label="fit_smart",
        init_psi=bad_psi_guess,
        init_psi_method="scan",
        parallel=False,
    )
    results_smart = dff.fits_df[fit_smart_guess.label]

    # Assert that the smart fit is significantly more accurate
    psi_error_bad = np.abs(results_bad["psi"].mean() - basic_sim_config.laser.psi)
    psi_error_smart = np.abs(results_smart["psi"].mean() - basic_sim_config.laser.psi)

    assert psi_error_smart < 0.1 * psi_error_bad
    assert psi_error_smart < 1e-5


# --- Tests for EKFFitter ---


def test_ekf_fitter_converges_on_static_signal(basic_sim_config):
    """
    Tests that the EKFFitter converges to the correct parameters for a
    static, noiseless signal.
    """
    dff = DeepFrame()
    dff.add_sim(basic_sim_config)
    dff.simulate(
        label="test_sim", n_seconds=50e-3, mode="snr", snr_db=80
    )  # A little noise helps EKF

    fit_obj = dff.fit(
        main_label="test_sim",
        method="ekf",
        # Use a higher process noise to allow for faster initial convergence
        Q_diag=[1e-6, 1e-6, 1e-5, 1e-5, 1e-6],
    )
    results = dff.fits_df[fit_obj.label]

    # Check the final converged values (last row of the results)
    final_values = results.iloc[-1]

    assert final_values["amp"] == pytest.approx(basic_sim_config.laser.amp, rel=1e-4)
    assert final_values["m"] == pytest.approx(basic_sim_config.m, rel=1e-4)
    compare_angles(final_values["phi"], basic_sim_config.ifo.phi, atol=1e-4)
    compare_angles(final_values["psi"], basic_sim_config.laser.psi, atol=1e-4)


def test_ekf_tracks_slowly_varying_phase(basic_sim_config):
    """
    Tests that the EKF can track a slowly changing phase signal.
    """
    dff = DeepFrame()

    n_seconds = 50e-3

    # To test the EKF's tracking, we need to create a signal where the
    # underlying phase (the ground truth) truly changes over time. We can
    # achieve this by setting the armlength modulation parameters.
    basic_sim_config.ifo.arml_mod_f = 0.2  # Very slow 0.2 Hz sine wave motion
    # Amplitude of motion in meters, chosen to produce ~2 rad of phase swing
    wl = basic_sim_config.laser.wavelength
    basic_sim_config.ifo.arml_mod_amp = 2.0 * wl / (4 * np.pi)

    dff.add_sim(basic_sim_config)
    # Use 'asd' mode to correctly generate the dynamic signal
    dff.simulate(label="test_sim", n_seconds=n_seconds, mode="asd", trial_num=123)

    fit_obj = dff.fit(main_label="test_sim", method="ekf")
    results = dff.fits_df[fit_obj.label]

    # Manually calculate the downsampled ground truth
    raw_data = dff.raws["test_sim"]
    n_cycles = fit_obj.n
    R, _, _ = _calculate_fit_params(raw_data, n_cycles)
    ground_truth_phi_downsamp = vectorized_downsample(raw_data.phi_sim, R)

    assert ground_truth_phi_downsamp is not None, "Downsampling of ground truth failed."

    # Check the RMS error between the tracked phase and the ground truth
    convergence_point = len(results) // 4
    tracked_phi = results["phi"].iloc[convergence_point:]
    true_phi = ground_truth_phi_downsamp[convergence_point:]

    # The EKF will have some lag, so we compare against a shifted ground truth
    # A simple way to account for lag is to compare the shape and amplitude.
    # A more robust test checks the RMS error.
    rms_error = np.sqrt(np.mean((tracked_phi - true_phi) ** 2))

    # The EKF isn't perfect, so we expect some tracking error.
    assert rms_error < 0.01

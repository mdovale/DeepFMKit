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
import pytest
from scipy.special import jv
from deepfmkit import fit
from deepfmkit.fit import ALL_PARAMS, DEFAULT_GUESS

# Mark all tests in this file as belonging to the 'fit' module
pytestmark = pytest.mark.fit


# A pytest fixture to provide a standard set of ground truth parameters
# and corresponding synthetic I/Q data for use in multiple tests.
@pytest.fixture
def synthetic_iq_data():
    """
    Provides a consistent set of ground truth parameters and the ideal,
    noiseless I/Q data that would be measured from such a system.
    """
    params = {
        "amp": 1.5,
        "m": 6.321,
        "phi": np.pi / 4,
        "psi": -np.pi / 8,
    }
    ndata = 20

    # --- Reference implementation of the I/Q model (from the manuscript) ---
    # This is a simple, clear, but slow version used to validate the fast one.
    q_model = np.zeros(ndata)
    i_model = np.zeros(ndata)
    n = np.arange(1, ndata + 1)

    # Note the model for Q_n and I_n from the paper corresponds to
    # the mean of the demodulated signals. The `fit` functions use a slightly
    # different convention internally for the Q and I data vectors passed to it.
    # We will use the internal convention for the test data.
    # From fit.py: model_q = C*cos(phi+npi/2)*J_n(m)*cos(npsi)
    #              model_i = -C*cos(phi+npi/2)*J_n(m)*sin(npsi)

    q_model = (
        params["amp"]
        * np.cos(params["phi"] + n * np.pi / 2)
        * jv(n, params["m"])
        * np.cos(n * params["psi"])
    )

    i_model = (
        -params["amp"]
        * np.cos(params["phi"] + n * np.pi / 2)
        * jv(n, params["m"])
        * np.sin(n * params["psi"])
    )

    iq_data = np.concatenate([q_model, i_model])

    param_vector = np.array([params[p] for p in ALL_PARAMS])

    return {"params": param_vector, "iq_data": iq_data, "ndata": ndata}


def test_ssqf_zero_at_ground_truth(synthetic_iq_data):
    """
    Tests that the SSQ (Sum of Squared Residuals) function `ssqf` returns
    a value very close to zero when evaluated with the true parameters that
    generated the data.
    """
    params = synthetic_iq_data["params"]
    iq_data = synthetic_iq_data["iq_data"]
    ndata = synthetic_iq_data["ndata"]

    ssq = fit.ssqf(ndata, iq_data, params)

    assert ssq == pytest.approx(0.0, abs=1e-12)


def test_coeffs_zero_gradient_at_ground_truth(synthetic_iq_data):
    """
    Tests that the `coeffs` function returns a gradient vector with all
    elements very close to zero when evaluated at the ground truth, which
    is a necessary condition for a minimum of the SSQ.
    """
    params = synthetic_iq_data["params"]
    iq_data = synthetic_iq_data["iq_data"]
    ndata = synthetic_iq_data["ndata"]

    ssq, jtj_flat, gradient = fit.coeffs(ndata, iq_data, params)

    # At the minimum of the SSQ, the gradient should be zero.
    assert ssq == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(gradient, np.zeros(4), atol=1e-9)


def test_fit_converges_to_ground_truth_from_good_guess(synthetic_iq_data):
    """
    This is the core validation test for the NLS fitter. It verifies that
    when started from a reasonably close initial guess, the `fit` function
    converges to the correct ground truth parameters.
    """
    true_params = synthetic_iq_data["params"]
    iq_data = synthetic_iq_data["iq_data"]
    ndata = synthetic_iq_data["ndata"]

    # Create an initial guess that is slightly perturbed from the truth
    initial_guess = true_params * np.array([0.9, 1.1, 1.2, 0.8])

    # Run the fitter
    status, final_params, final_ssq = fit.fit(ndata, iq_data, initial_guess)

    # --- Assertions ---
    assert status == 0, "Fit should report success (status 0)."
    assert final_ssq == pytest.approx(0.0, abs=1e-9), (
        "Final SSQ should be near zero for a perfect fit."
    )

    # Use a helper for comparing angles, which can wrap around
    def compare_angles(a1, a2, atol):
        diff = np.abs(a1 - a2)
        assert np.min([diff, 2 * np.pi - diff]) < atol

    # Compare the fitted parameters to the ground truth
    np.testing.assert_allclose(
        final_params[[0, 1]],
        true_params[[0, 1]],
        rtol=1e-6,
        err_msg="Amp or m did not converge.",
    )
    compare_angles(final_params[2], true_params[2], atol=1e-6)  # Check phi
    compare_angles(final_params[3], true_params[3], atol=1e-6)  # Check psi


def test_fit_with_constrained_parameters():
    """
    Tests that the fitter can correctly hold some parameters constant while
    fitting for others.
    """
    # FIX: Define the full set of parameters that will be used.
    # The fixed parameters will use the default values from the fitter.
    true_params_dict = {
        "amp": 1.8,
        "m": DEFAULT_GUESS["m"],  # Data is generated with the default m
        "phi": -0.7,
        "psi": DEFAULT_GUESS["psi"],  # Data is generated with the default psi
    }

    # Generate the synthetic I/Q data using these exact parameters.
    ndata = 20
    n = np.arange(1, ndata + 1)
    q_model = (
        true_params_dict["amp"]
        * np.cos(true_params_dict["phi"] + n * np.pi / 2)
        * jv(n, true_params_dict["m"])
        * np.cos(n * true_params_dict["psi"])
    )
    i_model = (
        -true_params_dict["amp"]
        * np.cos(true_params_dict["phi"] + n * np.pi / 2)
        * jv(n, true_params_dict["m"])
        * np.sin(n * true_params_dict["psi"])
    )
    iq_data = np.concatenate([q_model, i_model])

    # We will hold 'm' and 'psi' constant and only fit for 'amp' and 'phi'
    fit_params_to_use = ["amp", "phi"]

    # Provide a perturbed initial guess ONLY for the active parameters
    initial_guess_active = np.array(
        [true_params_dict["amp"] * 1.2, true_params_dict["phi"] * 0.7]
    )

    # Run the constrained fit. Now the model and data match perfectly.
    status, final_params_full, final_ssq = fit.fit(
        ndata, iq_data, initial_guess_active, fit_params=fit_params_to_use
    )

    # --- Assertions ---
    assert status == 0, (
        "Fit should now succeed because the model can perfectly match the data."
    )
    assert final_ssq == pytest.approx(0.0, abs=1e-9)

    # Check that the fitted parameters converged to the truth
    assert final_params_full[0] == pytest.approx(
        true_params_dict["amp"], rel=1e-6
    )  # amp
    assert final_params_full[2] == pytest.approx(
        true_params_dict["phi"], rel=1e-6
    )  # phi

    # Check that the fixed parameters are correctly set to the default values
    assert final_params_full[1] == pytest.approx(DEFAULT_GUESS["m"])  # m
    assert final_params_full[3] == pytest.approx(DEFAULT_GUESS["psi"])  # psi


def test_fit_robust_retry_mechanism():
    """
    Tests the `_find_best_initial_guess` fallback mechanism by providing a
    very poor initial guess that should cause the initial fit to fail,
    triggering the grid-search retry.
    """
    # Create synthetic data with a known 'm' value
    true_m = 15.3
    params = np.array([1.0, true_m, 0.5, 0.1])
    ndata = 20
    q_model = (
        params[0]
        * np.cos(params[2] + np.arange(1, ndata + 1) * np.pi / 2)
        * jv(np.arange(1, ndata + 1), params[1])
        * np.cos(np.arange(1, ndata + 1) * params[3])
    )
    i_model = (
        -params[0]
        * np.cos(params[2] + np.arange(1, ndata + 1) * np.pi / 2)
        * jv(np.arange(1, ndata + 1), params[1])
        * np.sin(np.arange(1, ndata + 1) * params[3])
    )
    iq_data = np.concatenate([q_model, i_model])

    # Provide a very bad initial guess, far from the true 'm'
    bad_initial_guess = np.array([1.5, 4.0, 0.0, 0.0])  # m=4.0 is far from m=15.3

    # Run the fit. We expect the first attempt to be bad, triggering the retry.
    # The retry should find a better guess and converge correctly.
    status, final_params, final_ssq = fit.fit(ndata, iq_data, bad_initial_guess)

    # --- Assertions ---
    # The status code '1' indicates a successful fit after a retry.
    assert status == 1
    assert final_ssq == pytest.approx(0.0, abs=1e-9)
    assert final_params[1] == pytest.approx(true_m, rel=1e-5)

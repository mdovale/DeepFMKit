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
from deepfmkit.core import DeepFrame
from deepfmkit.dsp import vectorized_downsample  # Import the helper

# Mark all tests in this file as belonging to the 'integration' module
pytestmark = pytest.mark.integration


# Helper function for comparing angles, accounting for wrapping at +/- pi
def compare_angles(a1, a2, atol):
    """Asserts that two angles are close, handling wrapping at 2*pi."""
    diff = np.abs(a1 - a2)
    shortest_distance = np.min([diff, 2 * np.pi - diff])
    assert shortest_distance < atol, (
        f"Angle difference {shortest_distance} exceeds tolerance {atol}"
    )


def test_full_workflow_noiseless_static(basic_sim_config):
    """
    Performs a full end-to-end integration test of the core workflow.
    """
    sim_config = basic_sim_config.copy()
    sim_label = "integration_test_sim"
    sim_config.label = sim_label

    ground_truth = {
        "m": sim_config.m,
        "phi": sim_config.ifo.phi,
        "psi": sim_config.laser.psi,
        "amp": sim_config.laser.amp,
        "dc": sim_config.laser.amp,
    }

    dff = DeepFrame()
    dff.add_sim(sim_config)
    dff.simulate(
        label=sim_label, n_seconds=10 / sim_config.laser.fm, mode="snr", snr_db=60
    )

    fit_label = "integration_test_fit"
    fit_obj = dff.fit(
        main_label=sim_label,
        method="nls",
        fit_label=fit_label,
        n=10,
        parallel=False,
        init_psi_method="scan",
    )

    assert fit_obj is not None
    results_df = dff.fits_df[fit_label]

    assert results_df["amp"].iloc[-1] == pytest.approx(ground_truth["amp"], rel=1e-4)
    assert results_df["m"].iloc[-1] == pytest.approx(ground_truth["m"], rel=1e-4)
    compare_angles(results_df["phi"].iloc[-1], ground_truth["phi"], atol=1e-4)
    compare_angles(results_df["psi"].iloc[-1], ground_truth["psi"], atol=1e-4)


def test_workflow_with_realistic_noise(basic_sim_config):
    """
    Tests that the simulation and fitting pipeline runs to completion with
    all noise sources enabled.
    """
    sim_config = basic_sim_config.copy()

    sim_config.laser.f_n = 100.0
    sim_config.laser.amp_n = 1e-4
    sim_config.laser.df_n = 1e3
    sim_config.ifo.arml_n = 1e-12
    sim_config.use_exact_physics = True

    dff = DeepFrame()
    dff.add_sim(sim_config)
    dff.simulate(label="test_sim", n_seconds=0.2, mode="asd", trial_num=42)

    fit_obj = dff.fit(main_label="test_sim", method="nls", parallel=False)

    assert fit_obj is not None
    results_df = dff.fits_df[fit_obj.label]

    assert np.all(np.isfinite(results_df["m"]))
    assert np.all(np.isfinite(results_df["phi"]))


def test_ekf_integration_on_dynamic_signal(basic_sim_config):
    """
    An integration test for the EKF fitter on a dynamic signal.
    """
    sim_config = basic_sim_config.copy()

    sim_config.ifo.arml_mod_f = 2.0
    sim_config.ifo.arml_mod_amp = 1e-9
    sim_config.use_exact_physics = True

    dff = DeepFrame()
    dff.add_sim(sim_config)
    dff.simulate(label="test_sim", n_seconds=2.0, mode="asd", trial_num=101)

    fit_obj = dff.fit(main_label="test_sim", method="ekf")

    assert fit_obj is not None

    # Manually downsample the ground truth for comparison
    raw_data = dff.raws["test_sim"]
    true_phi = vectorized_downsample(raw_data.phi_sim, fit_obj.R)

    results_df = dff.fits_df[fit_obj.label]
    convergence_point = len(results_df) // 4
    tracked_phi = results_df["phi"].iloc[convergence_point:]
    true_phi_stable = true_phi[convergence_point:]

    std_tracked = np.std(np.unwrap(tracked_phi))
    std_true = np.std(np.unwrap(true_phi_stable))

    assert std_tracked == pytest.approx(std_true, rel=0.3)


# To make SimConfig copyable, we need to add a copy method to it.
def add_copy_method_to_simconfig():
    """Adds a copy method to SimConfig to avoid test interference."""
    import copy
    from deepfmkit.physics import SimConfig

    def simconfig_copy(self):
        # Perform a deepcopy of the laser and ifo objects to ensure independence
        new_laser = copy.deepcopy(self.laser)
        new_ifo = copy.deepcopy(self.ifo)
        return SimConfig(
            label=self.label,
            laser_config=new_laser,
            ifo_config=new_ifo,
            f_samp=self.f_samp,
        )

    SimConfig.copy = simconfig_copy


add_copy_method_to_simconfig()

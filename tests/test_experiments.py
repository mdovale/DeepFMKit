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
import pickle
import gzip
import os
import pytest
from deepfmkit.experiments import Experiment
from deepfmkit.factories import ExperimentFactory, StandardDFMIExperimentFactory
from deepfmkit.physics import LaserConfig, IfoConfig
from deepfmkit.waveforms import shd

# Mark all tests in this file as belonging to the 'experiments' module
pytestmark = pytest.mark.experiments

# --- Mock Factory and Workers (DEFINED AT TOP LEVEL) ---


class MockFactory(ExperimentFactory):
    """A mock factory for testing. It does not need to log calls."""

    def __call__(self, params):
        return {"laser_config": LaserConfig(fm=1000.0), "main_ifo_config": IfoConfig()}

    def _get_expected_params_keys(self):
        return {"axis1", "axis2", "static1", "stoch1"}


def mock_run_single_trial_passthrough(job_packet):
    """
    Mock worker that returns the parameters it received.
    Must be a top-level function to be pickleable.
    """
    trial_params, _, _, _, _, _ = job_packet
    return {"point_params": trial_params, "results": {"received_params": trial_params}}


def mock_run_single_trial_aggregator(job_packet):
    """
    Mock worker that returns predictable numerical results for testing aggregation.
    Must be a top-level function to be pickleable.
    """
    trial_params, _, _, _, _, _ = job_packet
    axis1_val = trial_params.get("axis1", 0)
    axis2_val = trial_params.get("axis2", 0)
    fake_results = {
        "analysis1": {
            "m": axis1_val * 10,
            "phi": axis2_val + trial_params["_exp_trial_idx"],
        }
    }
    return {"point_params": trial_params, "results": fake_results}


# --- Tests for the Experiment Class ---


def test_experiment_job_generation_2d_sweep():
    """
    Tests that the Experiment class passes the correct parameters to the worker.
    """
    exp = Experiment(description="Test 2D Sweep")
    exp.set_config_factory(MockFactory())

    exp.add_axis("axis1", np.array([10, 20]))
    exp.add_axis("axis2", np.array([1, 2, 3]))
    exp.n_trials = 4
    exp.add_analysis(name="received_params", fitter_method="mock")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "deepfmkit.experiments._run_single_trial", mock_run_single_trial_passthrough
        )
        results = exp.run(n_cores=2)

    # Check the aggregated '_exp_point_idx' to verify job generation
    all_indices = results["received_params"]["_exp_point_idx"]["all_trials"]
    assert all_indices.shape == (2, 3, 4)

    # Check the indices of the first job: point (0,0), trial 0
    assert all_indices[0, 0, 0] == (0, 0)

    # Check the indices of the last job: point (1,2), trial 3
    assert all_indices[1, 2, 3] == (1, 2)

    # Check a value from another parameter
    all_axis1_vals = results["received_params"]["axis1"]["all_trials"]
    assert all_axis1_vals[1, 2, 3] == 20


def test_experiment_result_aggregation():
    """
    Tests that the run method correctly aggregates numerical results.
    """
    exp = Experiment(description="Test Aggregation")
    exp.set_config_factory(MockFactory())

    exp.add_axis("axis1", np.array([10, 20]))
    exp.add_axis("axis2", np.array([1, 2, 3]))
    exp.n_trials = 4
    exp.add_analysis(name="analysis1", fitter_method="mock")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "deepfmkit.experiments._run_single_trial", mock_run_single_trial_aggregator
        )
        results = exp.run(n_cores=2)

    assert results["analysis1"]["m"]["all_trials"].shape == (2, 3, 4)
    assert results["analysis1"]["phi"]["mean"].shape == (2, 3)

    m_slice_for_point = results["analysis1"]["m"]["all_trials"][1, 0, :]
    np.testing.assert_allclose(m_slice_for_point, 200)

    phi_slice_for_point = results["analysis1"]["phi"]["all_trials"][0, 2, :]
    np.testing.assert_allclose(phi_slice_for_point, [3, 4, 5, 6])


def test_stochastic_variable_generation():
    """
    Tests that stochastic variables are generated and passed correctly.
    """
    exp = Experiment(description="Test Stochastic", seed=42)
    exp.set_config_factory(MockFactory())
    exp.n_trials = 100

    def normal_generator():
        return np.random.normal(loc=5.0, scale=0.1)

    exp.add_stochastic_variable("stoch1", normal_generator)
    exp.set_static({"axis1": 1, "axis2": 1, "static1": 1})
    exp.add_analysis(name="received_params", fitter_method="mock")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "deepfmkit.experiments._run_single_trial", mock_run_single_trial_passthrough
        )
        results = exp.run(n_cores=2)

    generated_stoch_vars = results["received_params"]["stoch1"]["all_trials"].flatten()

    assert len(generated_stoch_vars) == 100
    assert np.mean(generated_stoch_vars) == pytest.approx(5.0, abs=0.1)
    assert np.std(generated_stoch_vars) == pytest.approx(0.1, rel=0.4)


def _compare_results_dicts(dict1, dict2, rtol=1e-7, atol=1e-9):
    """
    A recursive helper function to compare two nested dictionaries that may
    contain NumPy arrays.
    """
    assert dict1.keys() == dict2.keys(), "Result dictionaries have different keys."

    for key in dict1:
        item1, item2 = dict1[key], dict2[key]

        if isinstance(item1, dict):
            # Recurse into sub-dictionaries
            _compare_results_dicts(item1, item2, rtol, atol)
        elif isinstance(item1, np.ndarray):
            # Compare NumPy arrays
            assert item1.dtype == item2.dtype, f"Array dtype mismatch for key '{key}'"
            if np.issubdtype(item1.dtype, np.number):
                np.testing.assert_allclose(
                    item1,
                    item2,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Array values differ for key '{key}'",
                )
            else:
                np.testing.assert_array_equal(
                    item1, item2, err_msg=f"Object array values differ for key '{key}'"
                )
        else:
            # Compare other types
            assert item1 == item2, f"Values differ for key '{key}'"


def test_experiment_full_integration_with_golden_file():
    """
    A full integration test that runs a complex experiment and compares its
    output to a pre-computed "golden file".

    This test is marked as 'slow' and can be skipped during rapid development
    by running `pytest -m "not slow"`.

    It validates the end-to-end reproducibility of the entire framework,
    including the `Experiment` class orchestration, the physics engine,
    noise generation, and the NLS fitter.
    """
    # --- 1. Load the golden results file ---
    test_dir = os.path.dirname(__file__)
    golden_file_path = os.path.join(test_dir, "golden_results.gz")
    assert os.path.exists(golden_file_path), (
        "Golden file 'tests/golden_results.pkl' not found. Please generate it first."
    )

    with gzip.open(golden_file_path, "rb") as f:
        golden_results = pickle.load(f)

    # --- 2. Re-create the exact experiment configuration ---
    # Every parameter, especially the seed, must be identical.
    factory = StandardDFMIExperimentFactory(waveform_function=shd, opd_main=0.05)

    experiment = Experiment(description="2nd Harmonic Distortion", seed=29)
    experiment.set_config_factory(factory)
    experiment.n_trials = 500
    experiment.f_samp = 200e3
    experiment.n_fit_cycles_per_trial = 1

    experiment.add_axis("distortion_amp", np.linspace(0.0, 0.0006, 10))
    experiment.set_static({"m_main": 6.0, "phi": 0.0})
    experiment.add_stochastic_variable(
        "distortion_phase", lambda: np.random.uniform(-np.pi, np.pi)
    )

    experiment.add_analysis(
        name="NLS_Fit",
        fitter_method="nls",
        result_cols=["m"],
        fitter_kwargs={"ndata": 30},
    )

    # --- 3. Run the experiment again ---
    # This will be slow, which is why the test is marked accordingly.
    new_results = experiment.run(n_cores=4)

    # --- 4. Compare the new results with the golden file ---
    _compare_results_dicts(new_results, golden_results)

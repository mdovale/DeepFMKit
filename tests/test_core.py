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
import pytest
import os
from deepfmkit.core import DeepFrame
from deepfmkit.physics import LaserConfig, IfoConfig, SimConfig
from deepfmkit.data import RawData, FitData, open_txt_maybe_gzip

# Mark all tests in this file as belonging to the 'core' module
pytestmark = pytest.mark.core


# Use the shared basic_sim_config fixture from conftest.py
def test_deepframe_initialization(basic_sim_config):
    """
    Tests that DeepFrame can be initialized correctly with various arguments.
    """
    # 1. Test empty initialization
    dff = DeepFrame()
    assert isinstance(dff, DeepFrame)
    assert not dff.sims and not dff.raws and not dff.fits

    # 2. Test initialization with a SimConfig object
    dff_with_sim = DeepFrame(sim_obj=basic_sim_config)
    assert "test_sim" in dff_with_sim.sims
    assert dff_with_sim.sims["test_sim"] is basic_sim_config


def test_load_and_add_sim(basic_sim_config):
    """
    Tests the methods for adding simulation configurations to DeepFrame.
    """
    dff = DeepFrame()

    dff.add_sim(basic_sim_config)
    assert "test_sim" in dff.sims

    new_label = dff.new_sim(label="new_channel")
    assert new_label == "new_channel"
    assert "new_channel" in dff.sims
    assert isinstance(dff.sims["new_channel"], SimConfig)


def test_simulate_creates_rawdata(basic_sim_config):
    """
    Tests that `dff.simulate()` creates a correctly structured RawData object.
    """
    dff = DeepFrame()
    dff.add_sim(basic_sim_config)

    label = "test_sim"
    n_seconds = 0.1
    dff.simulate(label=label, n_seconds=n_seconds, mode="snr", snr_db=100)

    assert label in dff.raws
    raw_obj = dff.raws[label]

    assert isinstance(raw_obj, RawData)
    assert raw_obj.label == label
    assert raw_obj.sim is basic_sim_config
    assert len(raw_obj.data) == int(n_seconds * basic_sim_config.f_samp)


def test_fit_creates_fitdata(basic_sim_config):
    """
    Tests that `dff.fit()` creates a correctly structured FitData object.
    """
    dff = DeepFrame()
    dff.add_sim(basic_sim_config)
    dff.simulate(label="test_sim", n_seconds=0.2, mode="snr", snr_db=1000)

    fit_label = "test_fit"
    fit_obj = dff.fit(
        main_label="test_sim", method="nls", fit_label=fit_label, parallel=False
    )

    assert fit_label in dff.fits
    assert dff.fits[fit_label] is fit_obj
    assert isinstance(fit_obj, FitData)
    assert fit_obj.nbuf > 0
    assert len(fit_obj.amp) == fit_obj.nbuf


def test_create_witness_channel(basic_sim_config):
    """
    Tests the helper function for creating a witness channel.
    """
    dff = DeepFrame()
    main_label = "main_channel"
    witness_label = "witness_channel"

    basic_sim_config.label = main_label
    dff.add_sim(basic_sim_config)

    target_m_witness = 0.5
    dff.create_witness_channel(
        main_channel_label=main_label,
        witness_channel_label=witness_label,
        m_witness=target_m_witness,
    )

    assert witness_label in dff.sims
    witness_sim = dff.sims[witness_label]
    assert witness_sim.laser is basic_sim_config.laser
    assert witness_sim.m == pytest.approx(target_m_witness)


# --- Tests using static files ---


@pytest.fixture
def test_file_paths():
    """Provides the paths to the static test files."""
    # Assumes tests are run from the project root directory
    test_dir = os.path.join(os.path.dirname(__file__))
    return {
        "raw": os.path.join(test_dir, "raw_data.gz"),
        "fit": os.path.join(test_dir, "fit_data.gz"),
    }


def test_load_raw_from_file(test_file_paths):
    """
    Tests loading a raw data file from a static, version-controlled file.
    """
    raw_file = test_file_paths["raw"]
    assert os.path.exists(raw_file), "raw_data.gz not found in tests/ directory"

    # !!! IMPORTANT: UPDATE THESE VALUES to match the content of your test file !!!
    EXPECTED_NUM_CHANNELS_RAW = 1  # Check line 3 of raw_data.gz
    EXPECTED_T0_RAW = 20210818171519  # Check line 4
    EXPECTED_FSAMP_RAW = 30000.0  # Check line 5
    EXPECTED_FM_RAW = 400.0  # Check line 6
    EXPECTED_NUM_SAMPLES_RAW = (
        304500  # This depends (but is not equal to) the number of data rows
    )

    dff = DeepFrame(raw_file=raw_file)

    # Assert header values were parsed correctly
    assert dff.channr == EXPECTED_NUM_CHANNELS_RAW
    assert dff.t0 == EXPECTED_T0_RAW
    assert dff.f_samp == pytest.approx(EXPECTED_FSAMP_RAW)
    assert dff.fm == pytest.approx(EXPECTED_FM_RAW)

    # Assert RawData object was created correctly
    # The default label is based on the filename
    default_label = f"{raw_file}_ch0"
    assert default_label in dff.raws
    raw_obj = dff.raws[default_label]
    assert len(raw_obj.data) == EXPECTED_NUM_SAMPLES_RAW


def test_load_fit_from_file(test_file_paths):
    """
    Tests loading a fit data file from a static, version-controlled file.
    """
    fit_file = test_file_paths["fit"]
    assert os.path.exists(fit_file), "fit_data.gz not found in tests/ directory"

    # !!! IMPORTANT: UPDATE THESE VALUES to match the content of your test file !!!
    EXPECTED_NUM_CHANNELS_FIT = 1  # Check line 3 of fit_data.gz
    EXPECTED_T0_FIT = 20210818171519  # Check line 4
    EXPECTED_N_FIT = 20  # Check line 7
    EXPECTED_FS_FIT = 20.0  # Check line 9
    EXPECTED_NUM_BUFFERS = 203  # This is the number of fit buffers in the entire file

    dff = DeepFrame()
    dff.load_fit(fit_file=fit_file)

    # Assert header values were parsed correctly
    assert dff.channr == EXPECTED_NUM_CHANNELS_FIT
    assert dff.t0 == EXPECTED_T0_FIT
    assert dff.n == EXPECTED_N_FIT
    assert dff.fs == pytest.approx(EXPECTED_FS_FIT)

    # Assert FitData object was created correctly
    default_label = f"{fit_file}_ch0"
    assert default_label in dff.fits
    fit_obj = dff.fits[default_label]
    assert fit_obj.nbuf == EXPECTED_NUM_BUFFERS
    assert len(fit_obj.amp) == EXPECTED_NUM_BUFFERS


def test_to_txt_saves_fit(basic_sim_config, tmp_path, test_file_paths):
    """
    Tests that the `to_txt` method correctly saves a FitData object by
    comparing its content to a known good file.
    """
    dff = DeepFrame()

    # Load a known fit object to save
    dff.load_fit(fit_file=test_file_paths["fit"], labels=["fit_to_save"])

    # Save the file to a temporary directory
    save_path = tmp_path / "output_dir"
    save_path.mkdir()
    dff.to_txt(filepath=str(save_path) + os.path.sep, labels=["fit_to_save"])

    # Check if the file was created
    output_file = save_path / "fit_to_save.txt"
    assert output_file.exists()

    # Compare the newly saved file with the original static test file.
    # We ignore the first two lines (timestamped message) and compare the rest.
    with open(output_file, "r") as f_new, open_txt_maybe_gzip(test_file_paths["fit"]) as f_orig:
        lines_new = f_new.readlines()
        lines_orig = f_orig.readlines()

        # Compare headers (skipping message line) and data
        assert lines_new[0] == lines_orig[0]  # '% fit_data'
        assert lines_new[2:] == lines_orig[2:]


def test_fit_init_raises_error_for_short_data(basic_sim_config):
    """
    Tests that dff.fit_init() raises a ValueError if the data is too short
    for the requested number of cycles per buffer.
    """
    sim = SimConfig(
        label="test_sim",
        laser_config=LaserConfig(),
        ifo_config=IfoConfig(),
        f_samp=200000.0,
    )
    dff = DeepFrame(sim)
    dff.simulate(label="test_sim", n_seconds=1 / basic_sim_config.laser.fm)

    # Now try to fit with n=20, which is impossible
    with pytest.raises(ValueError):
        dff.fit(main_label="test_sim", n=20)

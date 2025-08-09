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
import scipy.constants as sc
from deepfmkit.physics import LaserConfig, IfoConfig, SimConfig, SignalGenerator
from deepfmkit.data import RawData

pytestmark = pytest.mark.physics


# A fixture to create a default, clean, noiseless simulation configuration
# that can be used as a starting point for many tests.
@pytest.fixture
def basic_sim_config() -> SimConfig:
    """Provides a basic, noiseless, static SimConfig object."""
    ifo = IfoConfig(label="test_ifo", ref_arml=0.1, meas_arml=0.3, phi=np.pi / 4)
    laser = LaserConfig(label="test_laser", fm=1000.0, amp=1.5, psi=-np.pi / 8)
    # Set df to achieve a known modulation index m
    laser.set_df_for_m(ifo, m=6.321)

    sim = SimConfig(
        label="test_sim", laser_config=laser, ifo_config=ifo, f_samp=200000.0
    )
    # Use the simplified model for tests that compare against the analytical formula
    sim.use_exact_physics = False
    return sim


def test_laser_config_set_df_for_m():
    """
    Tests that the `set_df_for_m` helper method correctly calculates the
    required frequency modulation amplitude `df`.
    """
    ifo = IfoConfig(ref_arml=0.1, meas_arml=0.3)  # OPD = 0.2 m
    laser = LaserConfig()

    m_target = 6.0
    opd = 0.2

    laser.set_df_for_m(ifo, m=m_target)

    # Theoretical formula: df = (m * c) / (2 * pi * OPD)
    expected_df = (m_target * sc.c) / (2 * np.pi * opd)

    assert laser.df == pytest.approx(expected_df)


def test_sim_config_m_property(basic_sim_config):
    """
    Tests that the read-only `m` property of SimConfig is calculated correctly
    and updates when underlying parameters change.
    """
    # 1. Check initial calculation
    opd = abs(basic_sim_config.ifo.meas_arml - basic_sim_config.ifo.ref_arml)
    expected_m = (2 * np.pi * basic_sim_config.laser.df * opd) / sc.c
    assert basic_sim_config.m == pytest.approx(expected_m)

    # 2. Check that it updates when a parameter changes
    basic_sim_config.laser.df *= 2
    assert basic_sim_config.m == pytest.approx(expected_m * 2)


def test_generate_ideal_signal_matches_formula(basic_sim_config):
    """
    This is a critical integration test. It verifies that the SignalGenerator's
    output for a simple case matches the known analytical formula for a DFMI signal.
    """
    n_seconds = 0.1
    sg = SignalGenerator()

    # Generate the signal using the simplified physics model
    result = sg.generate(main_config=basic_sim_config, n_seconds=n_seconds, mode="asd")
    generated_signal = result["main"].data["ch0"].to_numpy()

    # Calculate the expected signal from the analytical formula
    cfg = basic_sim_config
    n_samples = int(n_seconds * cfg.f_samp)
    t = np.arange(n_samples) / cfg.f_samp

    # v(t) = A * (1 + V * cos(phi + m * cos(2*pi*fm*t + psi)))
    # For this test, DC offset A is cfg.laser.amp, and AC amplitude C=A*V is also cfg.laser.amp
    # because visibility is 1.0. The formula in the manuscript is v(t) = DC + AC*cos(...)
    # The simulation produces v(t) = (DC+noise_amp)*(1 + V*cos(...)). For V=1, DC=AC.
    # The term `laser.amp` in the simulation corresponds to the DC offset.
    dc_offset = cfg.laser.amp
    ac_amplitude = cfg.laser.amp * cfg.ifo.visibility

    phase_mod = cfg.m * np.cos(2 * np.pi * cfg.laser.fm * t + cfg.laser.psi)
    total_phase = cfg.ifo.phi + phase_mod
    expected_signal = dc_offset + ac_amplitude * np.cos(total_phase)

    np.testing.assert_allclose(generated_signal, expected_signal, atol=1e-9)


@pytest.mark.parametrize(
    "noise_attr, value",
    [
        ("f_n", 100.0),  # Laser frequency noise
        ("amp_n", 0.1),  # Amplitude noise
        ("arml_n", 1e-12),  # Arm length noise
    ],
)
def test_noise_injection_increases_variance(basic_sim_config, noise_attr, value):
    """
    Tests that enabling a noise source increases the output signal's variance,
    confirming that the noise is being correctly injected.
    """
    n_seconds = 0.1
    sg = SignalGenerator()

    # 1. Generate baseline noiseless signal
    result_noiseless = sg.generate(
        main_config=basic_sim_config, n_seconds=n_seconds, mode="asd", trial_num=1
    )
    variance_noiseless = np.var(result_noiseless["main"].data["ch0"])

    # 2. Enable the specified noise source
    if hasattr(basic_sim_config.laser, noise_attr):
        setattr(basic_sim_config.laser, noise_attr, value)
    elif hasattr(basic_sim_config.ifo, noise_attr):
        setattr(basic_sim_config.ifo, noise_attr, value)

    # 3. Generate noisy signal
    result_noisy = sg.generate(
        main_config=basic_sim_config, n_seconds=n_seconds, mode="asd", trial_num=1
    )
    variance_noisy = np.var(result_noisy["main"].data["ch0"])

    # 4. Assert that noise made a difference
    assert variance_noisy > variance_noiseless


def test_dynamic_signal_creates_sidebands(basic_sim_config):
    """
    Tests that enabling dynamic arm length modulation creates the expected
    phase modulation sidebands in the frequency spectrum of the signal.
    """
    n_seconds = 1.0
    sg = SignalGenerator()

    # Configure a dynamic motion
    f_mod_arml = 21.0  # Use a non-integer frequency to avoid harmonic overlap
    basic_sim_config.ifo.arml_mod_f = f_mod_arml
    basic_sim_config.ifo.arml_mod_amp = 5e-9  # 5 nm motion

    # Generate the dynamic signal
    result = sg.generate(main_config=basic_sim_config, n_seconds=n_seconds, mode="asd")
    signal = result["main"].data["ch0"].to_numpy()

    # Analyze the spectrum
    fs = basic_sim_config.f_samp
    n_samples = len(signal)
    fft_result = np.abs(np.fft.rfft(signal))
    fft_freqs = np.fft.rfftfreq(n_samples, 1 / fs)

    # Find the bin indices for the carrier (fm) and a sideband
    fm_carrier_idx = np.argmin(np.abs(fft_freqs - basic_sim_config.laser.fm))
    sideband_idx = np.argmin(
        np.abs(fft_freqs - (basic_sim_config.laser.fm + f_mod_arml))
    )
    # Check a nearby empty bin for comparison
    empty_bin_idx = np.argmin(
        np.abs(fft_freqs - (basic_sim_config.laser.fm + 3 * f_mod_arml))
    )

    carrier_power = fft_result[fm_carrier_idx]
    sideband_power = fft_result[sideband_idx]
    noise_floor_power = fft_result[empty_bin_idx]

    # Assert that the sideband power is significant compared to the carrier
    assert sideband_power > 0.01 * carrier_power
    # Assert that the sideband is well above the local noise floor
    assert sideband_power > 1e4 * noise_floor_power


def test_exact_physics_converges_to_simple_model_large_m(basic_sim_config):
    """
    Tests that the high-fidelity physics model (with timeshift) converges to
    the simplified analytical model in the limit of very small time delays.
    """
    n_seconds = 0.05
    sg = SignalGenerator()

    # Modify the config for a very small OPD, which makes tau small
    basic_sim_config.ifo.meas_arml = basic_sim_config.ifo.ref_arml + 0.1  # 10 cm OPD
    basic_sim_config.laser.set_df_for_m(basic_sim_config.ifo, m=6.0)

    # 1. Generate with exact physics
    basic_sim_config.use_exact_physics = True
    result_exact = sg.generate(
        main_config=basic_sim_config, n_seconds=n_seconds, mode="asd"
    )
    signal_exact = result_exact["main"].data["ch0"].to_numpy()

    # 2. Generate with simplified physics
    basic_sim_config.use_exact_physics = False
    result_simple = sg.generate(
        main_config=basic_sim_config, n_seconds=n_seconds, mode="asd"
    )
    signal_simple = result_simple["main"].data["ch0"].to_numpy()

    # 3. Assert they are very close
    np.testing.assert_allclose(signal_exact, signal_simple, rtol=1e-3)


def test_exact_physics_converges_to_simple_model_small_m(basic_sim_config):
    """
    Tests that the high-fidelity physics model (with timeshift) converges to
    the simplified analytical model in the limit of very small time delays.
    """
    n_seconds = 0.05
    sg = SignalGenerator()

    # Modify the config for a very small OPD, which makes tau small
    basic_sim_config.ifo.meas_arml = basic_sim_config.ifo.ref_arml + 0.01  # 1 cm OPD
    basic_sim_config.laser.set_df_for_m(basic_sim_config.ifo, m=0.1)

    # 1. Generate with exact physics
    basic_sim_config.use_exact_physics = True
    result_exact = sg.generate(
        main_config=basic_sim_config, n_seconds=n_seconds, mode="asd"
    )
    signal_exact = result_exact["main"].data["ch0"].to_numpy()

    # 2. Generate with simplified physics
    basic_sim_config.use_exact_physics = False
    result_simple = sg.generate(
        main_config=basic_sim_config, n_seconds=n_seconds, mode="asd"
    )
    signal_simple = result_simple["main"].data["ch0"].to_numpy()

    # 3. Assert they are very close
    np.testing.assert_allclose(signal_exact, signal_simple, rtol=1e-7)


def test_witness_channel_generation(basic_sim_config):
    """
    Tests that the generator can create a main and a witness channel simultaneously.
    """
    sg = SignalGenerator()

    # Create a separate witness config sharing the same laser
    witness_ifo = IfoConfig(label="witness_ifo", ref_arml=0.01, meas_arml=0.015)
    witness_sim = SimConfig(
        label="witness", laser_config=basic_sim_config.laser, ifo_config=witness_ifo
    )

    # Generate both channels
    results = sg.generate(
        main_config=basic_sim_config,
        witness_config=witness_sim,
        n_seconds=0.05,
        mode="asd",
    )

    # Assert the structure of the output
    assert isinstance(results, dict)
    assert "main" in results
    assert "witness" in results
    assert isinstance(results["main"], RawData)
    assert isinstance(results["witness"], RawData)
    assert len(results["main"].data) == len(results["witness"].data)

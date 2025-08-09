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
from deepfmkit.experiments import Experiment
from deepfmkit.waveforms import shd
from deepfmkit.factories import StandardDFMIExperimentFactory
import numpy as np
import pickle
import gzip
import os

# This script generates the reference data for `test_experiment_full_integration_with_golden_file`
# It should only be run once to create the 'tests/golden_results.pkl' file.


def generate():
    print("Generating golden file for experiment tests...")

    factory = StandardDFMIExperimentFactory(waveform_function=shd, opd_main=0.05)

    experiment = Experiment(description="2nd Harmonic Distortion", seed=29)
    experiment.set_config_factory(factory)
    experiment.n_trials = 500
    experiment.f_samp = 200e3
    # Corrected variable name to match the class attribute
    experiment.n_fit_cycles_per_trial = 1

    axis = np.linspace(0.0, 0.0006, 10)
    experiment.add_axis("distortion_amp", axis)

    experiment.set_static(
        {
            "m_main": 6.0,
            "phi": 0.0,
        }
    )

    np.random.seed(experiment.seed)  # Ensure the lambda is deterministic

    experiment.add_stochastic_variable(
        "distortion_phase", lambda: np.random.uniform(-np.pi, np.pi)
    )

    experiment.add_analysis(
        name="NLS_Fit",
        fitter_method="nls",
        result_cols=["m"],
        fitter_kwargs={"ndata": 30},
    )

    print(f"Starting experiment: '{experiment.description}'...")
    # Use a smaller number of cores to avoid potential memory issues with large trials
    results = experiment.run(n_cores=4)
    print("Experiment completed.")

    # Save the results to a pickle file inside the tests directory
    output_path = os.path.join("tests", "golden_results.gz")
    with gzip.open(output_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Golden file saved successfully to: {output_path}")


if __name__ == "__main__":
    generate()

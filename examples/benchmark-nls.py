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
import time
import os
import subprocess
import numpy as np
from deepfmkit.core import DeepFrame
from deepfmkit.physics import LaserConfig, IfoConfig, SimConfig

def get_cpu_name():
    """Gets the specific CPU brand string for the current machine."""
    try:
        # This command is specific to macOS
        if os.uname().sysname == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).strip().decode()
        # TODO: Add equivalent commands for Linux/Windows
        # elif os.uname().sysname == "Linux":
        #     # ...
    except Exception:
        return "Unknown CPU"
    return "Unknown CPU"


def setup_benchmark(n_seconds, f_samp, fm):
    """
    Creates the DFF instance and simulates the data needed for the benchmark.
    This setup is done once to ensure we are only timing the fitting process.
    """
    print("--- Setting up benchmark environment ---")
    print(f"Simulating {n_seconds}s of data at {f_samp/1e3:.0f} kS/s...")
    
    dff = DeepFrame()
    
    # Use a standard, simple configuration
    ifo = IfoConfig(ref_arml=0.1, meas_arml=0.3)
    laser = LaserConfig(fm=fm)
    laser.set_df_for_m(ifo, m=6.0)
    
    sim_config = SimConfig(
        label="benchmark_data",
        laser_config=laser,
        ifo_config=ifo,
        f_samp=f_samp
    )
    dff.add_sim(sim_config)
    
    # Simulate data with a high SNR to ensure the fit is well-behaved
    dff.simulate(label="benchmark_data", n_seconds=n_seconds, mode='snr', snr_db=100)
    
    print("Setup complete.\n")
    return dff, "benchmark_data"


def run_timing(fit_function, repetitions=5):
    """
    Runs a function multiple times and returns the minimum execution time.
    
    Using the minimum time is a standard practice in benchmarking, as it is
    less susceptible to outliers caused by other system processes.
    
    A warm-up run is performed first to account for any initial JIT compilation
    or caching effects.
    """
    # Warm-up run
    _ = fit_function()
    
    times = []
    for _ in range(repetitions):
        start_time = time.perf_counter()
        fit_function()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
    return min(times)


if __name__ == "__main__":
    # --- Benchmark Parameters (from manuscript) ---
    N_SECONDS = 100.0
    F_SAMP = 200e3
    FM = 1e3
    N_CYCLES_PER_BUFFER = 20  # This is the 'n' parameter
    N_HARMONICS = 30          # This is the 'n_harmonics' parameter
    N_REPETITIONS = 5         # Number of times to run each test for stable timing

    # --- Derived Parameters ---
    # The number of fits is determined by the data length and buffer size
    n_samples = int(N_SECONDS * F_SAMP)
    samples_per_buffer = int(F_SAMP / FM * N_CYCLES_PER_BUFFER)
    n_fits = n_samples // samples_per_buffer
    
    print(f"Benchmark Configuration:")
    print(f"  - Total Data Duration: {N_SECONDS}s")
    print(f"  - Cycles per Fit (n): {N_CYCLES_PER_BUFFER}")
    print(f"  - Harmonics per Fit (n_harmonics): {N_HARMONICS}")
    print(f"  - Total Number of Fits to Perform: {n_fits}")
    print("-" * 40)
    
    dff, data_label = setup_benchmark(n_seconds=N_SECONDS, f_samp=F_SAMP, fm=FM)
    
    # --- 1. Sequential Timing ---
    print(f"Running SEQUENTIAL benchmark ({N_REPETITIONS} repetitions)...")
    seq_fit_func = lambda: dff.fit(
        main_label=data_label,
        method='nls',
        n=N_CYCLES_PER_BUFFER,
        n_harmonics=N_HARMONICS,
        parallel=False  # Explicitly disable parallelism
    )
    seq_time = run_timing(seq_fit_func, repetitions=N_REPETITIONS)
    print(f"Best sequential time: {seq_time:.4f} seconds\n")

    # --- 2. Parallel Timing ---
    n_cores = os.cpu_count()
    print(f"Running PARALLEL benchmark on {n_cores} cores ({N_REPETITIONS} repetitions)...")
    par_fit_func = lambda: dff.fit(
        main_label=data_label,
        method='nls',
        n=N_CYCLES_PER_BUFFER,
        n_harmonics=N_HARMONICS,
        parallel=True,  # Enable parallelism
        n_cores=n_cores
    )
    par_time = run_timing(par_fit_func, repetitions=N_REPETITIONS)
    print(f"Best parallel time: {par_time:.4f} seconds\n")
    
    # --- 3. Results ---
    speedup = seq_time / par_time
    cpu_name = get_cpu_name()
    
    print("=" * 40)
    print("           BENCHMARK RESULTS")
    print("=" * 40)
    print(f"Processor:          {cpu_name}")
    print(f"Sequential Time (s): {seq_time:.2f}")
    print(f"Parallel Time (s):   {par_time:.2f}")
    print(f"Speedup Factor:     {speedup:.1f}x")
    print("-" * 40)
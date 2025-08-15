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
        if os.name == 'posix':
            if os.uname().sysname == "Darwin":
                return subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"]
                ).strip().decode()
            elif os.uname().sysname == "Linux":
                with open('/proc/cpuinfo') as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(':')[1].strip()
    except Exception:
        return "Unknown CPU"
    return "Unknown CPU"

def setup_benchmark_data(n_seconds, f_samp, fm, label="benchmark_data"):
    """
    Creates a DFF instance and simulates data needed for a benchmark.
    """
    print(f"--- Setting up data for '{label}' ---")
    print(f"Simulating {n_seconds}s of data at {f_samp/1e3:.0f} kS/s...")
    
    dff = DeepFrame()
    
    ifo = IfoConfig(ref_arml=0.1, meas_arml=0.3)
    laser = LaserConfig(fm=fm)
    laser.set_df_for_m(ifo, m=6.0)
    
    sim_config = SimConfig(
        label=label,
        laser_config=laser,
        ifo_config=ifo,
        f_samp=f_samp
    )
    dff.add_sim(sim_config)
    dff.simulate(label=label, n_seconds=n_seconds, mode='snr', snr_db=100)
    
    print("Setup complete.\n")
    return dff, label

def run_timing(fit_function, repetitions=5):
    """
    Runs a function multiple times and returns the minimum execution time.
    A warm-up run is performed first.
    """
    print(f"Performing warm-up run...")
    _ = fit_function()
    
    times = []
    print(f"Running benchmark {repetitions} times...")
    for i in range(repetitions):
        start_time = time.perf_counter()
        fit_function()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        print(f"  Run {i+1}/{repetitions}: {times[-1]:.4f} s")
        
    return min(times)

if __name__ == "__main__":
    cpu_name = get_cpu_name()
    print(f"Starting benchmarks on processor: {cpu_name}")
    print("=" * 60)

    # --- Common Parameters ---
    F_SAMP = 200e3
    FM = 1e3
    REPETITIONS = 3
    N_SECONDS = 100.0
    n_samples = int(N_SECONDS * F_SAMP)

    # --- NLS Fitter Benchmark ---
    print("\n           NLS FITTER BENCHMARK")
    print("-" * 60)
    N_CYCLES_PER_BUFFER = 20
    N_HARMONICS = 30

    samples_per_buffer = int(F_SAMP / FM * N_CYCLES_PER_BUFFER)
    n_fits = n_samples // samples_per_buffer
    
    print(f"Configuration:")
    print(f"  - Data Duration: {N_SECONDS}s ({n_samples} samples)")
    print(f"  - Total Number of Fits: {n_fits}")
    print(f"  - Cycles per Fit (n): {N_CYCLES_PER_BUFFER}")
    print(f"  - Harmonics per Fit (n_harmonics): {N_HARMONICS}")
    
    dff_nls, data_label_nls = setup_benchmark_data(
        n_seconds=N_SECONDS, f_samp=F_SAMP, fm=FM, label="nls_data"
    )
    
    # 1. NLS Sequential Timing
    print("\nRunning NLS SEQUENTIAL benchmark...")
    seq_fit_func = lambda: dff_nls.fit(
        main_label=data_label_nls, method='nls', n=N_CYCLES_PER_BUFFER,
        n_harmonics=N_HARMONICS, parallel=False, verbose=False
    )
    seq_time = run_timing(seq_fit_func, repetitions=REPETITIONS)
    nls_seq_throughput_ksps = (n_samples / seq_time) / 1000.0
    print(f"-> Best sequential time: {seq_time:.4f} seconds")
    print(f"-> NLS Sequential Throughput: {nls_seq_throughput_ksps:.1f} kS/s\n")

    # 2. NLS Parallel Timing
    n_cores = os.cpu_count()
    print(f"Running NLS PARALLEL benchmark on {n_cores} cores...")
    par_fit_func = lambda: dff_nls.fit(
        main_label=data_label_nls, method='nls', n=N_CYCLES_PER_BUFFER,
        n_harmonics=N_HARMONICS, parallel=True, n_cores=n_cores, verbose=False
    )
    par_time = run_timing(par_fit_func, repetitions=REPETITIONS)
    nls_par_throughput_ksps = (n_samples / par_time) / 1000.0
    print(f"-> Best parallel time: {par_time:.4f} seconds")
    print(f"-> NLS Parallel Throughput: {nls_par_throughput_ksps:.1f} kS/s\n")

    # --- EKF Fitter Benchmark (Random Walk) ---
    print("\n           EKF FITTER BENCHMARK (Random Walk)")
    print("-" * 60)
    print(f"Configuration:")
    print(f"  - Data Duration: {N_SECONDS}s ({n_samples} samples)")
    
    dff_ekf, data_label_ekf = setup_benchmark_data(
        n_seconds=N_SECONDS, f_samp=F_SAMP, fm=FM, label="ekf_data"
    )

    print("\nRunning EKF benchmark...")
    ekf_fit_func = lambda: dff_ekf.fit(main_label=data_label_ekf, method='ekf', verbose=False)
    ekf_time = run_timing(ekf_fit_func, repetitions=REPETITIONS)
    
    ekf_throughput_ksps = (n_samples / ekf_time) / 1000.0
    print(f"-> Best EKF time: {ekf_time:.4f} seconds")
    print(f"-> EKF Throughput: {ekf_throughput_ksps:.1f} kS/s\n")

    # --- Integrated EKF Fitter Benchmark ---
    print("\n       INTEGRATED EKF FITTER BENCHMARK (Constant Velocity)")
    print("-" * 60)
    print(f"Configuration:")
    print(f"  - Data Duration: {N_SECONDS}s ({n_samples} samples)")
    
    dff_iekf, data_label_iekf = setup_benchmark_data(
        n_seconds=N_SECONDS, f_samp=F_SAMP, fm=FM, label="iekf_data"
    )

    print("\nRunning Integrated EKF benchmark...")
    iekf_fit_func = lambda: dff_iekf.fit(main_label=data_label_iekf, method='iekf', verbose=False)
    iekf_time = run_timing(iekf_fit_func, repetitions=REPETITIONS)
    
    iekf_throughput_ksps = (n_samples / iekf_time) / 1000.0
    print(f"-> Best Integrated EKF time: {iekf_time:.4f} seconds")
    print(f"-> Integrated EKF Throughput: {iekf_throughput_ksps:.1f} kS/s\n")


    # --- Final Results Summary ---
    speedup = seq_time / par_time
    
    print("=" * 60)
    print("                 FINAL BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Processor:          {cpu_name}")
    print("\n--- NLS Fitter ---")
    print(f"Sequential Time:    {seq_time:.2f} s (for {n_fits} fits on {N_SECONDS:.0f}s of data)")
    print(f"Parallel Time:      {par_time:.2f} s ({n_cores} cores)")
    print(f"Speedup Factor:     {speedup:.1f}x")
    print(f"Throughput (Seq):   {nls_seq_throughput_ksps:.1f} kS/s")
    print(f"Throughput (Par):   {nls_par_throughput_ksps:.1f} kS/s")
    print("\n--- EKF Fitters ---")
    print(f"EKF (Random Walk) Time:          {ekf_time:.2f} s (for {N_SECONDS:.0f}s of data)")
    print(f"EKF (Random Walk) Throughput:    {ekf_throughput_ksps:.1f} kS/s")
    print(f"EKF (Const. Vel.) Time:          {iekf_time:.2f} s (for {N_SECONDS:.0f}s of data)")
    print(f"EKF (Const. Vel.) Throughput:    {iekf_throughput_ksps:.1f} kS/s")
    print("-" * 60)
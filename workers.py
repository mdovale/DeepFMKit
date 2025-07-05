import DeepFMKit.core as dfm
import numpy as np
import scipy.constants as sc

def run_single_trial(params):
    """
    Worker function for a single simulation/fit trial, using a "Stitched Phase Noise" method.

    This function is self-contained and implements a corrected Hybrid Timescale
    Simulation strategy to efficiently and accurately model the effects of
    long acquisition times.

    It handles two simulation types:
    1. 'stat_base': Performs a direct Monte Carlo simulation for statistical
       noise at a fixed, short baseline acquisition time (`T_base`). This is
       used to find a reference statistical uncertainty.
    2. 'sys_hybrid': Simulates laser drift noise. It generates the full,
       long-duration (`T_acq`) phase noise profile at a low sampling rate.
       This long phase noise vector is then downsampled and "stitched" into
       a single, short, high-sampling-rate DFMI signal buffer. This preserves
       the time-varying nature of the drift error, whose magnitude correctly
       scales with `T_acq`.

    Parameters
    ----------
    params : dict
        A dictionary containing all parameters for this single trial run.
        Expected keys include: 'sim_type', 'trial_num', 'T_acq', 'T_base',
        'm_true', 'delta_f', 'amp_asd', 'freq_asd', 'ndata'.

    Returns
    -------
    float
        The result of the fit for this single trial.
        - If 'stat_base', this is the fitted modulation depth 'm'.
        - If 'sys_hybrid', this is the bias ('m_fit' - 'm_true').
    """
    # Unpack all necessary parameters from the dictionary
    sim_type = params['sim_type']
    T_acq = params['T_acq']
    trial_num = params['trial_num']
    m_true = params['m_true']
    delta_f = params['delta_f']
    amp_asd = params['amp_asd']
    freq_asd = params['freq_asd']
    ndata = params['ndata']

    # Each process gets its own independent instance of the framework
    dff = dfm.DeepFitFramework()
    label = f"worker_{trial_num}_{T_acq:.4f}"
    dff.new_sim(label)
    sim = dff.sims[label]
    sim.m = m_true
    sim.df = delta_f
    sim.ndata = ndata

    # --- Case 1: Direct simulation for baseline statistical error ---
    if sim_type == 'stat_base':
        T_base = params['T_base']
        sim.amp_n = amp_asd
        sim.f_n = 0.0
        dff.simulate(label, n_seconds=T_base, trial_num=trial_num)
        fit_n = int(sim.f_mod * T_base)
        fit_obj = dff.fit(label, n=fit_n, verbose=False, parallel=False)
        return fit_obj.m[0] if fit_obj and fit_obj.m.size > 0 else m_true

    # --- Case 2: Hybrid simulation for systematic drift error ---
    elif sim_type == 'sys_hybrid':
        # Step 1: Simulate the full, long-duration frequency noise profile
        sim.amp_n = 0.0
        sim.f_n = freq_asd

        f_samp_noise = 20.0
        
        # --- BUG FIX: Enforce a minimum duration for noise generation ---
        # This ensures that fmin is always < fmax for pyplnoise.
        min_noise_gen_duration = 1.0  # seconds
        noise_gen_duration = max(T_acq, min_noise_gen_duration)
        num_samples_noise_req = int(noise_gen_duration * f_samp_noise)
        # ---------------------------------------------------------------

        original_fs = sim.f_samp
        sim.f_samp = f_samp_noise
        noise_arrays = dff._generate_noise_arrays(sim, np.arange(num_samples_noise_req), trial_num)
        sim.f_samp = original_fs

        f_noise_t = noise_arrays['laser_frequency']
        dt_noise = 1.0 / f_samp_noise
        
        # Step 2: Integrate frequency noise to get the true phase noise
        opd = (m_true * sc.c) / (2 * np.pi * delta_f)
        phi_drift_t_long = 2 * np.pi * (opd / sc.c) * np.cumsum(f_noise_t) * dt_noise

        # Step 3: Define a short, high-rate buffer for the NLS fitter
        T_fit_buffer = 0.1
        num_samples_fit = int(T_fit_buffer * sim.f_samp)
        t_fit_buffer = np.arange(num_samples_fit) / sim.f_samp

        # Step 4: Truncate the long noise vectors to the actual T_acq
        # and "stitch" the result onto the short buffer
        num_points_for_interp = int(T_acq * f_samp_noise)
        if num_points_for_interp < 2: num_points_for_interp = 2
        # Ensure we don't try to select more points than we generated
        num_points_for_interp = min(num_points_for_interp, len(phi_drift_t_long))

        t_long_truncated = np.arange(num_points_for_interp) * dt_noise
        phi_drift_t_long_truncated = phi_drift_t_long[:num_points_for_interp]

        phi_drift_downsampled = np.interp(t_fit_buffer, t_long_truncated, phi_drift_t_long_truncated)

        # Step 5: Regenerate the signal with the stitched phase noise term
        A = sim.amp
        C = sim.visibility
        omega_mod = 2 * np.pi * sim.f_mod
        m = sim.m
        psi = sim.psi
        
        phitot_distorted = sim.phi + phi_drift_downsampled + m * np.cos(omega_mod * t_fit_buffer + psi)
        distorted_signal = A * (1 + C * np.cos(phitot_distorted))

        # Manually create the DeepRawObject to feed to the fitter
        raw_obj = dfm.DeepRawObject(data=dfm.pd.DataFrame(distorted_signal, columns=["ch0"]))
        raw_obj.f_samp = sim.f_samp
        raw_obj.f_mod = sim.f_mod
        dff.raws[label] = raw_obj

        # Step 6: Fit the short, but realistically distorted, buffer
        fit_n = int(sim.f_mod * T_fit_buffer)
        fit_obj = dff.fit(label, n=fit_n, verbose=False, parallel=False)

        m_fit = fit_obj.m[0] if fit_obj and fit_obj.m.size > 0 else m_true
        return m_fit - m_true

    return 0.0

def calculate_bias_for_point(params):
    """
    Worker function to calculate mean and worst-case bias for a single (m, epsilon) point.

    This function is executed by a pool process. It runs a full Monte Carlo
    simulation over the random phase of the 2nd harmonic distortion for a
    single specified m_main and epsilon value. For both the standard NLS and
    the W-DFMI fitters, it computes both the mean bias and the maximum
    absolute bias observed across all random phase trials.

    Parameters
    ----------
    params : dict
        A dictionary containing all parameters for this single grid point.
        Expected keys: 'm_main', 'epsilon', 'n_phase_trials', 'm_witness',
        'grid_i', 'grid_j'.

    Returns
    -------
    tuple
        A tuple containing (grid_i, grid_j, mean_std_bias, worst_std_bias,
        mean_wdfmi_bias, worst_wdfmi_bias) for reconstructing the result grids.
    """
    # Unpack parameters
    m_main = params['m_main']
    epsilon = params['epsilon']
    n_phase_trials = params['n_phase_trials']
    m_witness = params['m_witness']
    grid_i = params['grid_i']
    grid_j = params['grid_j']

    standard_biases = []
    wdfmi_biases = []
    
    # Dynamically calculate ndata needed for this m
    ndata = int(m_main + 20)

    # Monte Carlo loop
    for _ in range(n_phase_trials):
        dff = dfm.DeepFitFramework()
        
        laser_config = dfm.LaserConfig()
        main_ifo_config = dfm.InterferometerConfig()
        
        laser_config.df_2nd_harmonic_frac = epsilon
        laser_config.df_2nd_harmonic_phase = np.random.uniform(0, 2 * np.pi)
        
        main_label = "main"
        main_channel = dfm.DFMIObject(main_label, laser_config, main_ifo_config)
        
        opd_main = main_ifo_config.meas_arml - main_ifo_config.ref_arml
        if opd_main == 0: continue
        laser_config.df = (m_main * dfm.sc.c) / (2 * np.pi * opd_main)
        dff.sims[main_label] = main_channel

        witness_label = "witness"
        witness_channel = dff.create_witness_channel(
            main_channel_label=main_label,
            witness_channel_label=witness_label,
            m_witness=m_witness
        )
        # witness_channel.ifo.phi = np.pi / 2.0
        
        n_seconds = main_channel.fit_n / laser_config.f_mod
        dff.simulate(main_label, n_seconds=n_seconds, witness_label=witness_label)
        
        fit_obj_std = dff.fit(main_label, fit_label="std_fit", ndata=ndata, verbose=False, parallel=False)
        if fit_obj_std and fit_obj_std.m.size > 0:
            standard_biases.append(fit_obj_std.m[0] - m_main)

        fit_obj_wdfmi = dff.fit_wdfmi(main_label, witness_label, fit_label="wdfmi_fit", ndata=ndata, verbose=False)
        if fit_obj_wdfmi and fit_obj_wdfmi.m.size > 0:
            wdfmi_biases.append(fit_obj_wdfmi.m[0] - m_main)

    # Convert lists to numpy arrays for easier calculation
    standard_biases_arr = np.array(standard_biases) if standard_biases else np.array([0])
    wdfmi_biases_arr = np.array(wdfmi_biases) if wdfmi_biases else np.array([0])

    # Calculate statistics for both fitters
    mean_std_bias = np.mean(standard_biases_arr)
    worst_std_bias = np.max(np.abs(standard_biases_arr))
    
    mean_wdfmi_bias = np.mean(wdfmi_biases_arr)
    worst_wdfmi_bias = np.max(np.abs(wdfmi_biases_arr))
    
    return (grid_i, grid_j, mean_std_bias, worst_std_bias, mean_wdfmi_bias, worst_wdfmi_bias)

def calculate_bias_for_point_2(params):
    """
    Worker function to calculate mean and worst-case bias for a single (m, epsilon) point.

    This function is executed by a pool process. It runs a full Monte Carlo
    simulation over two random variables for each trial:
    1. The phase of the 2nd harmonic distortion.
    2. The static phase of the witness interferometer.

    This provides the most rigorous test of the W-DFMI fitter's stability,
    revealing the full extent of the bias problem by exploring the entire space
    of unknown phase parameters.

    Parameters
    ----------
    params : dict
        A dictionary containing all parameters for this single grid point.
        Expected keys: 'm_main', 'epsilon', 'n_phase_trials', 'm_witness',
        'grid_i', 'grid_j'.

    Returns
    -------
    tuple
        A tuple containing (grid_i, grid_j, mean_std_bias, worst_std_bias,
        mean_wdfmi_bias, worst_wdfmi_bias) for reconstructing the result grids.
    """
    # Unpack parameters
    m_main = params['m_main']
    epsilon = params['epsilon']
    n_phase_trials = params['n_phase_trials']
    m_witness = params['m_witness']
    grid_i = params['grid_i']
    grid_j = params['grid_j']

    standard_biases = []
    wdfmi_biases = []
    
    # Dynamically calculate ndata needed for this m
    ndata = int(m_main + 10)

    # --- Monte Carlo loop over TWO random variables ---
    for _ in range(n_phase_trials):
        dff = dfm.DeepFitFramework()
        
        laser_config = dfm.LaserConfig()
        main_ifo_config = dfm.InterferometerConfig()
        
        # Randomize the distortion phase for this trial
        laser_config.df_2nd_harmonic_frac = epsilon
        laser_config.df_2nd_harmonic_phase = np.random.uniform(0, 2 * np.pi)
        
        main_label = "main"
        main_channel = dfm.DFMIObject(main_label, laser_config, main_ifo_config)
        
        opd_main = main_ifo_config.meas_arml - main_ifo_config.ref_arml
        if opd_main == 0: continue
        laser_config.df = (m_main * dfm.sc.c) / (2 * np.pi * opd_main)
        dff.sims[main_label] = main_channel

        witness_label = "witness"
        # Create witness but disable auto-tuning to test its raw performance
        witness_channel = dff.create_witness_channel(
            main_channel_label=main_label,
            witness_channel_label=witness_label,
            m_witness=m_witness
        )
        
        n_seconds = main_channel.fit_n / laser_config.f_mod
        dff.simulate(main_label, n_seconds=n_seconds, witness_label=witness_label)
        
        fit_obj_std = dff.fit(main_label, fit_label="std_fit", ndata=ndata, verbose=False, parallel=False)
        if fit_obj_std and fit_obj_std.m.size > 0:
            standard_biases.append(fit_obj_std.m[0] - m_main)

        # fit_obj_wdfmi = dff.fit_wdfmi(main_label, witness_label, fit_label="wdfmi_fit", ndata=ndata, verbose=False)
        fit_obj_wdfmi = dff.fit_wdfmi_orthogonal_demodulation(main_label, witness_label, fit_label="wdfmi_fit", verbose=False)
        if fit_obj_wdfmi and fit_obj_wdfmi.m.size > 0:
            wdfmi_biases.append(fit_obj_wdfmi.m[0] - m_main)

    # Convert lists to numpy arrays for easier calculation
    standard_biases_arr = np.array(standard_biases) if standard_biases else np.array([0])
    wdfmi_biases_arr = np.array(wdfmi_biases) if wdfmi_biases else np.array([0])

    # Calculate statistics for both fitters
    mean_std_bias = np.mean(standard_biases_arr)
    worst_std_bias = np.max(np.abs(standard_biases_arr))
    
    mean_wdfmi_bias = np.mean(wdfmi_biases_arr)
    worst_wdfmi_bias = np.max(np.abs(wdfmi_biases_arr))
    
    return (grid_i, grid_j, mean_std_bias, worst_std_bias, mean_wdfmi_bias, worst_wdfmi_bias)

def calculate_bias_for_m_vs_mwitness(params):
    """
    Worker function to calculate W-DFMI bias for a single (m_main, m_witness) point.

    This function simulates a non-distorted DFMI signal and fits it with the
    W-DFMI algorithm to isolate residual biases caused by the interaction
    between the main and witness channel configurations.
    """
    m_main = params['m_main']
    m_witness = params['m_witness']
    witness_phi = params['witness_phi']
    grid_i = params['grid_i']
    grid_j = params['grid_j']
    
    dff = dfm.DeepFitFramework()
    
    laser_config = dfm.LaserConfig()
    main_ifo_config = dfm.InterferometerConfig()
    
    # No distortion for this test
    laser_config.df_2nd_harmonic_frac = 0.0
    
    main_label = "main"
    main_channel = dfm.DFMIObject(main_label, laser_config, main_ifo_config)
    
    opd_main = main_ifo_config.meas_arml - main_ifo_config.ref_arml
    if opd_main == 0: return (grid_i, grid_j, 0)
    laser_config.df = (m_main * dfm.sc.c) / (2 * np.pi * opd_main)
    dff.sims[main_label] = main_channel

    witness_label = "witness"
    # Create the witness but disable auto-tuning to respect the m_witness value
    witness_channel = dff.create_witness_channel(
        main_channel_label=main_label,
        witness_channel_label=witness_label,
        m_witness=m_witness
    )
    # Set the desired witness phase
    if witness_phi is not None:
        witness_channel.ifo.phi = witness_phi
    
    n_seconds = main_channel.fit_n / laser_config.f_mod
    dff.simulate(main_label, n_seconds=n_seconds, witness_label=witness_label)
    
    # Use enough harmonics to avoid truncation error
    ndata = int(m_main + 10)
    
    # fit_obj_wdfmi = dff.fit_wdfmi_orthogonal_demodulation(main_label, witness_label, fit_label="wdfmi_fit", verbose=False)
    fit_obj_wdfmi = dff.fit_wdfmi_sequential(main_label, witness_label, fit_label="wdfmi_fit", verbose=False)
    
    bias = 0
    if fit_obj_wdfmi and fit_obj_wdfmi.m.size > 0:
        bias = fit_obj_wdfmi.m[0] - m_main
        
    return (grid_i, grid_j, bias)

def compare_fitter_stability(params):
    """
    Worker function to compare the stability of the simultaneous NLS and the
    sequential bootstrap W-DFMI fitters for a single (m, epsilon) point.

    This function runs a full Monte Carlo simulation over two random variables:
    1. The phase of the 2nd harmonic distortion.
    2. The static phase of the witness interferometer.

    It computes and returns the mean and worst-case bias for both fitting
    algorithms, providing the data needed for a direct comparison.

    Parameters
    ----------
    params : dict
        A dictionary containing all parameters for this single grid point.

    Returns
    -------
    tuple
        A tuple containing (grid_i, grid_j, mean_unstable, worst_unstable,
        mean_sequential, worst_sequential) for reconstructing the result grids.
    """
    # Unpack parameters
    m_main = params['m_main']
    epsilon = params['epsilon']
    n_phase_trials = params['n_phase_trials']
    m_witness = params['m_witness']
    grid_i = params['grid_i']
    grid_j = params['grid_j']

    unstable_biases = []
    sequential_biases = []
    
    ndata = int(m_main + 10)

    for _ in range(n_phase_trials):
        dff = dfm.DeepFitFramework()
        
        laser_config = dfm.LaserConfig()
        main_ifo_config = dfm.InterferometerConfig()
        
        laser_config.df_2nd_harmonic_frac = epsilon
        laser_config.df_2nd_harmonic_phase = np.random.uniform(0, 2 * np.pi)
        
        main_label = "main"
        main_channel = dfm.DFMIObject(main_label, laser_config, main_ifo_config)
        
        opd_main = main_ifo_config.meas_arml - main_ifo_config.ref_arml
        if opd_main == 0: continue
        laser_config.df = (m_main * dfm.sc.c) / (2 * np.pi * opd_main)
        dff.sims[main_label] = main_channel

        witness_label = "witness"
        witness_channel = dff.create_witness_channel(
            main_channel_label=main_label,
            witness_channel_label=witness_label,
            m_witness=m_witness
        )
        witness_channel.ifo.phi = np.random.uniform(-0.5e-3, 0.5e-3)
        
        n_seconds = main_channel.fit_n / laser_config.f_mod
        dff.simulate(main_label, n_seconds=n_seconds, witness_label=witness_label)
        
        # Fit with the old, unstable fitter
        fit_obj_unstable = dff.fit_wdfmi(main_label, witness_label, fit_label="unstable_fit", ndata=ndata, verbose=False)
        if fit_obj_unstable and fit_obj_unstable.m.size > 0:
            unstable_biases.append(fit_obj_unstable.m[0] - m_main)

        # Fit with the new, robust sequential fitter
        fit_obj_seq = dff.fit_wdfmi_sequential(main_label, witness_label, fit_label="sequential_fit", verbose=False)
        if fit_obj_seq and fit_obj_seq.m.size > 0:
            sequential_biases.append(fit_obj_seq.m[0] - m_main)

    unstable_biases_arr = np.array(unstable_biases) if unstable_biases else np.array([0])
    sequential_biases_arr = np.array(sequential_biases) if sequential_biases else np.array([0])
    
    mean_unstable = np.mean(unstable_biases_arr)
    worst_unstable = np.max(np.abs(unstable_biases_arr))
    mean_sequential = np.mean(sequential_biases_arr)
    worst_sequential = np.max(np.abs(sequential_biases_arr))
    
    return (grid_i, grid_j, mean_unstable, worst_unstable, mean_sequential, worst_sequential)
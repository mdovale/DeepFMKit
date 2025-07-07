import DeepFMKit.core as dfm
import numpy as np
import scipy.constants as sc

def calculate_ambiguity_boundary_point(params):
    """
    Calculates the coarse phase error for a single point in the
    (delta_f, delta_m) parameter space.

    This is a simple, non-simulation worker that applies the core theoretical
    equation: delta_Phi_coarse = -delta_m * (f0 / df).

    Parameters
    ----------
    params : dict
        A dictionary containing the point's parameters:
        - 'delta_f': The modulation amplitude (df).
        - 'delta_m': The total uncertainty/bias on the m estimate.
        - 'f0': The laser carrier frequency.
        - 'grid_i', 'grid_j': Indices for the output grid.

    Returns
    -------
    tuple
        (grid_i, grid_j, coarse_phase_error)
    """
    # This worker is purely computational, no need for DFF.
    delta_f = params['delta_f']
    delta_m = params['delta_m']
    f0 = params['f0']
    grid_i = params['grid_i']
    grid_j = params['grid_j']
    
    # Avoid division by zero if delta_f is zero
    if delta_f == 0:
        return (grid_i, grid_j, float('inf'))
        
    coarse_phase_error = -delta_m * (f0 / delta_f)
    
    return (grid_i, grid_j, np.abs(coarse_phase_error))


def run_efficiency_trial(params):
    """
    A self-contained worker for a single fitter efficiency trial.
    
    This version is corrected to ensure each trial uses a DFMIObject with a
    unique label, preventing the "Invalid label" error during fitting.
    """
    # Self-contained imports
    import DeepFMKit.core as dfm
    import numpy as np

    # Unpack all parameters from the input dictionary
    trial_num = params['trial_num']
    # We now pass the config objects directly
    laser_config = params['laser_config']
    ifo_config = params['ifo_config']
    n_seconds = params['n_seconds']
    snr_db = params['snr_db']
    ndata = params['ndata']
    m_true = params['m_true']
    
    # Each worker gets its own DFF instance
    dff_worker = dfm.DeepFitFramework()
    
    # --- The Fix is Here ---
    # I now create a NEW DFMIObject for each trial with a unique label.
    # This ensures the generated raw data object has the correct label.
    label = f"eff_trial_{trial_num}"
    sim_config = dfm.DFMIObject(label, laser_config, ifo_config)
    dff_worker.sims[label] = sim_config
    # --- End of Fix ---
    
    # Simulate a single noisy buffer
    dff_worker.simulate(
        label,
        n_seconds=n_seconds,
        mode='snr',
        snr_db=snr_db,
        trial_num=trial_num
    )
    
    # Fit the buffer using the standard NLS fitter
    fit_obj = dff_worker.fit(label, ndata=ndata, init_m=m_true, verbose=False, parallel=False)
    
    return fit_obj.m[0] if fit_obj and fit_obj.m.size > 0 else np.nan

def calculate_single_distortion_bias(params):
    """
    Worker to calculate the bias for a single trial with modulation distortion.

    This function simulates one instance of a distorted signal, where the
    distortion amplitude (epsilon) and phase are specified. It then fits the
    signal with both a W-DFMI fitter and the conventional DFMI fitter,
    returning the bias from each.

    This is the core computational unit for the Monte Carlo analysis of
    systematic errors from modulation non-linearity.

    Parameters
    ----------
    params : dict
        A dictionary containing all parameters for this single trial.
        - 'm_main': Target modulation depth.
        - 'm_witness': Fixed modulation depth for the witness channel.
        - 'distortion_amp': Amplitude of the 2nd harmonic distortion (epsilon).
        - 'distortion_phase': Phase of the 2nd harmonic distortion.
        - 'wdfmi_fitter_name': String name of the W-DFMI fitter to use.

    Returns
    -------
    tuple
        A tuple (wdfmi_bias, dfmi_bias).
    """
    # Self-contained imports
    import DeepFMKit.core as dfm
    import numpy as np

    # Unpack parameters
    m_main_target = params['m_main']
    m_witness_target = params['m_witness']
    distortion_amp = params['distortion_amp']
    distortion_phase = params['distortion_phase']
    wdfmi_fitter_name = params['wdfmi_fitter_name']

    dff = dfm.DeepFitFramework()

    # 1. Configure the simulation
    laser_config = dfm.LaserConfig()
    main_ifo_config = dfm.InterferometerConfig()
    
    laser_config.df_2nd_harmonic_frac = distortion_amp
    laser_config.df_2nd_harmonic_phase = distortion_phase

    opd_main = main_ifo_config.meas_arml - main_ifo_config.ref_arml
    if opd_main == 0: return np.nan, np.nan
    laser_config.df = (m_main_target * dfm.sc.c) / (2 * np.pi * opd_main)
    
    main_label = "main"
    main_channel = dfm.DFMIObject(main_label, laser_config, main_ifo_config)
    dff.sims[main_label] = main_channel

    # 2. Design the witness IFO
    witness_label = "witness"
    opd_witness = (m_witness_target * dfm.sc.c) / (2 * np.pi * laser_config.df)
    witness_ifo_config = dfm.InterferometerConfig(label="witness_ifo")
    witness_ifo_config.ref_arml = 0.01
    witness_ifo_config.meas_arml = witness_ifo_config.ref_arml + opd_witness
    
    f0 = dfm.sc.c / laser_config.wavelength
    static_fringe_phase = (2 * np.pi * f0 * opd_witness) / dfm.sc.c
    witness_ifo_config.phi = (np.pi / 2.0) - static_fringe_phase
    
    witness_channel = dfm.DFMIObject(witness_label, laser_config, witness_ifo_config, f_samp=main_channel.f_samp)
    dff.sims[witness_label] = witness_channel

    # 3. Simulate and Fit
    n_seconds = main_channel.fit_n / laser_config.f_mod
    dff.simulate(main_label, n_seconds=n_seconds, witness_label=witness_label, mode='asd')
    
    # --- Run W-DFMI Fitter ---
    wdfmi_bias = np.nan
    try:
        wdfmi_fitter_func = getattr(dff, wdfmi_fitter_name)
        fit_wdfmi = wdfmi_fitter_func(main_label, witness_label, verbose=False)
        if fit_wdfmi and fit_wdfmi.m.size > 0:
            wdfmi_bias = fit_wdfmi.m[0] - m_main_target
    except Exception:
        pass

    # --- Run Conventional DFMI Fitter ---
    dfmi_bias = np.nan
    try:
        fit_dfmi = dff.fit(main_label, verbose=False, parallel=False)
        if fit_dfmi and fit_dfmi.m.size > 0:
            dfmi_bias = fit_dfmi.m[0] - m_main_target
    except Exception:
        pass

    return wdfmi_bias, dfmi_bias

def calculate_stat_sys_tradeoff(params):
    """
    Modernized worker for analyzing the statistical vs. systematic error tradeoff.

    This function is a refactoring of the old `run_single_trial` worker, updated
    to use the current, more robust DeepFMKit codebase. It supports two modes:
    1. 'stat': A standard Monte Carlo run to find the statistical uncertainty of
       the `m` parameter for a given acquisition time.
    2. 'sys_hybrid': Implements the "Hybrid Timescale Simulation" to find the
       systematic bias on `m` from laser frequency drift over a long
       acquisition time, `T_acq`.

    Key Improvements:
    - Uses the proper `DFMIObject` configuration instead of setting `m` directly.
    - Leverages the exact physics model from the `SignalGenerator` class for
      signal reconstruction, avoiding first-order approximations.
    - Includes the corrected logic for the hybrid timescale interpolation.

    Parameters
    ----------
    params : dict
        A dictionary containing all parameters for this single trial run.

    Returns
    -------
    float
        - If 'stat', returns the fitted modulation depth `m_fit`.
        - If 'sys_hybrid', returns the bias (`m_fit` - `m_true`).
    """
    # Self-contained imports for parallel processing
    import DeepFMKit.core as dfm
    import numpy as np

    # Unpack parameters
    sim_type = params['sim_type']
    T_acq = params['T_acq']
    m_true = params['m_true']

    # --- 1. Configure the simulation based on physical parameters ---
    dff = dfm.DeepFitFramework()
    label = f"worker_{sim_type}_{T_acq:.4f}"
    
    laser_config = dfm.LaserConfig()
    ifo_config = dfm.InterferometerConfig()

    opd = ifo_config.meas_arml - ifo_config.ref_arml
    laser_config.df = (m_true * dfm.sc.c) / (2 * np.pi * opd)

    if sim_type == 'stat':
        laser_config.amp_n = params['amp_asd']
        laser_config.f_n = 0.0
    else: # 'sys_hybrid'
        laser_config.amp_n = 0.0
        laser_config.f_n = params['freq_asd']

    main_channel = dfm.DFMIObject(label, laser_config, ifo_config)
    dff.sims[label] = main_channel

    # --- 2. Execute the appropriate simulation and fitting logic ---
    if sim_type == 'stat':
        dff.simulate(label, n_seconds=T_acq, mode='asd', trial_num=params['trial_num'])
        fit_n = int(main_channel.laser.f_mod * T_acq)
        fit_obj = dff.fit(label, n=fit_n, init_m=m_true, verbose=False, parallel=False)
        return fit_obj.m[0] if fit_obj and fit_obj.m.size > 0 else m_true

    elif sim_type == 'sys_hybrid':
        # Step 1: Generate the long-duration, low-rate frequency noise
        f_samp_noise = 20.0
        noise_gen_duration = max(T_acq, 1.0)
        num_samples_noise_req = int(noise_gen_duration * f_samp_noise)
        
        original_fs = main_channel.f_samp
        main_channel.f_samp = int(f_samp_noise)
        generator_for_noise = dfm.SignalGenerator()
        noise_arrays = generator_for_noise._generate_noise_arrays(main_channel, np.arange(num_samples_noise_req), params['trial_num'])
        main_channel.f_samp = original_fs

        f_noise_t = noise_arrays['laser_frequency']
        dt_noise = 1.0 / f_samp_noise

        # Step 2: Integrate to get the true phase noise profile
        phi_drift_t_long = 2 * np.pi * (opd / dfm.sc.c) * np.cumsum(f_noise_t) * dt_noise

        # Step 3: Define a short, high-rate buffer for the NLS fitter
        T_fit_buffer = 0.1
        num_samples_fit = int(T_fit_buffer * main_channel.f_samp)
        t_fit_buffer = np.arange(num_samples_fit) / main_channel.f_samp

        # Step 4: "Stitch" the long noise profile onto the short buffer
        num_points_for_interp = min(int(T_acq * f_samp_noise), len(phi_drift_t_long))
        if num_points_for_interp < 2: num_points_for_interp = 2
        
        # --- THE FIX IS HERE ---
        # The source time axis must be rescaled to match the duration of the
        # short fit buffer. This "compresses" the long noise profile into the
        # short buffer, preserving its statistical magnitude (variance).
        t_source_rescaled = np.linspace(0, T_fit_buffer, num=num_points_for_interp)
        phi_drift_stitched = np.interp(t_fit_buffer, t_source_rescaled, phi_drift_t_long[:num_points_for_interp])
        # --- END OF FIX ---

        # Step 5: Manually reconstruct the signal using the EXACT physics model
        omega_mod = 2 * np.pi * laser_config.f_mod
        dt_fit = 1.0 / main_channel.f_samp
        
        g_t_normalized = np.cos(omega_mod * t_fit_buffer + laser_config.psi)
        phi_mod_waveform = 2 * np.pi * laser_config.df * np.cumsum(g_t_normalized) * dt_fit
        
        tau_r = ifo_config.ref_arml / dfm.sc.c
        tau_m = ifo_config.meas_arml / dfm.sc.c
        
        phi_mod_meas = np.interp(t_fit_buffer - tau_m, t_fit_buffer, phi_mod_waveform)
        phi_mod_ref = np.interp(t_fit_buffer - tau_r, t_fit_buffer, phi_mod_waveform)
        delta_phi_mod = phi_mod_meas - phi_mod_ref
        
        omega_0_clean = 2 * np.pi * dfm.sc.c / laser_config.wavelength
        delta_phi_carrier = omega_0_clean * (tau_m - tau_r)

        phitot_distorted = delta_phi_carrier + delta_phi_mod + phi_drift_stitched
        distorted_signal = laser_config.amp * (1 + laser_config.visibility * np.cos(phitot_distorted))
        
        raw_obj = dfm.DeepRawObject(data=dfm.pd.DataFrame(distorted_signal, columns=["ch0"]))
        raw_obj.f_samp = main_channel.f_samp
        raw_obj.f_mod = main_channel.laser.f_mod
        dff.raws[label] = raw_obj
        
        # Step 6: Fit the short but distorted buffer
        fit_n = int(main_channel.laser.f_mod * T_fit_buffer)
        fit_obj = dff.fit(label, n=fit_n, init_m=m_true, verbose=False, parallel=False)
        
        m_fit = fit_obj.m[0] if fit_obj and fit_obj.m.size > 0 else m_true
        return m_fit - m_true
    
    return np.nan

def calculate_bias_for_point(params):
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

    This function simulates a non-distorted DFMI signal and fits it with a
    specified W-DFMI algorithm. It is designed to respect the physical
    constraints of the W-DFMI method by ensuring the witness interferometer's
    modulation depth (m_witness) is held constant, even as the laser's
    modulation amplitude (df) changes to achieve the target m_main.

    The key logic is:
    1. Determine the required laser df for the target m_main.
    2. Design the witness interferometer's physical path length (delta_l) to
       achieve the target m_witness with that specific df.
    3. Analytically lock the witness interferometer to its quadrature point.
    4. Simulate the system and perform the fit.

    Parameters
    ----------
    params : dict
        A dictionary containing all parameters for the simulation point:
        - 'm_main': The target modulation depth for the main channel.
        - 'm_witness': The target modulation depth for the witness channel.
        - 'fitter_func_name': The string name of the fitting function to use
                              (e.g., 'fit_wdfmi_orthogonal_demodulation').
        - 'grid_i', 'grid_j': Indices for placing the result in the output grid.

    Returns
    -------
    tuple
        A tuple (grid_i, grid_j, bias) with the calculated bias value.
    """
    # I've set this up to be completely self-contained for parallel processing.
    import DeepFMKit.core as dfm
    import numpy as np

    # Unpack parameters
    m_main_target = params['m_main']
    m_witness_target = params['m_witness']
    fitter_func_name = params['fitter_func_name']
    grid_i = params['grid_i']
    grid_j = params['grid_j']

    dff = dfm.DeepFitFramework()

    # --- 1. Configure the shared laser and main interferometer ---
    laser_config = dfm.LaserConfig()
    main_ifo_config = dfm.InterferometerConfig(label="main_ifo")
    
    # Ensure there is no signal distortion for this intrinsic bias test
    laser_config.df_2nd_harmonic_frac = 0.0
    
    # Calculate the required laser modulation amplitude (df) to achieve the
    # target m_main with the fixed main interferometer path length.
    opd_main = main_ifo_config.meas_arml - main_ifo_config.ref_arml
    if opd_main == 0: return (grid_i, grid_j, 0)
    laser_config.df = (m_main_target * dfm.sc.c) / (2 * np.pi * opd_main)
    
    main_label = "main"
    main_channel = dfm.DFMIObject(main_label, laser_config, main_ifo_config)
    dff.sims[main_label] = main_channel

    # --- 2. Design the Witness Interferometer to meet the Golden Rule ---
    # The witness modulation depth (m_witness) must be maintained at its target
    # value. To do this, I must adjust its physical path difference (OPD)
    # to compensate for the changing laser df.
    witness_label = "witness"
    
    # Calculate the required physical path difference for the witness
    opd_witness = (m_witness_target * dfm.sc.c) / (2 * np.pi * laser_config.df)

    # Now, create a new IFO config with this specific OPD
    witness_ifo_config = dfm.InterferometerConfig(label="witness_ifo")
    witness_ifo_config.ref_arml = 0.01  # A small, non-zero base armlength
    witness_ifo_config.meas_arml = witness_ifo_config.ref_arml + opd_witness

    # With the physical structure defined, I can now perform the analytical fringe lock.
    # This sets witness_ifo_config.phi to place the IFO at quadrature.
    f0 = dfm.sc.c / laser_config.wavelength
    static_fringe_phase = (2 * np.pi * f0 * opd_witness) / dfm.sc.c
    witness_ifo_config.phi = (np.pi / 2.0) - static_fringe_phase

    # Finally, compose the witness DFMIObject
    witness_channel = dfm.DFMIObject(witness_label, laser_config, witness_ifo_config, f_samp=main_channel.f_samp)
    dff.sims[witness_label] = witness_channel

    # --- 3. Simulate and Fit ---
    n_seconds = main_channel.fit_n / laser_config.f_mod
    dff.simulate(main_label, n_seconds=n_seconds, witness_label=witness_label, mode='asd')
    
    # Retrieve the specified fitter function from the framework object
    try:
        fitter_function = getattr(dff, fitter_func_name)
    except AttributeError:
        # Handle case where the function name is invalid
        return (grid_i, grid_j, np.nan) 

    # Execute the chosen fitter
    if fitter_func_name == 'fit':
        fit_obj = fitter_function(main_label, fit_label="wdfmi_fit", ndata=int(m_main_target+15), parallel=False, verbose=False)
    else:
        fit_obj = fitter_function(main_label, witness_label, fit_label="wdfmi_fit", verbose=False)
    
    bias = np.nan
    if fit_obj and fit_obj.m.size > 0:
        # The true value is m_main_target
        bias = fit_obj.m[0] - m_main_target
        
    return (grid_i, grid_j, bias)

def calculate_bias_for_m_vs_mwitness_stochastic(params):
    """
    Worker function for a stochastic analysis of W-DFMI bias.

    For a single (m_main, m_witness) grid point, this function runs a
    Monte Carlo simulation over multiple trials. In each trial, the witness
    interferometer's phase is perturbed by a random amount from its ideal
    quadrature point. This simulates an imperfectly locked witness and tests
    the robustness of the fitting algorithm.

    It returns the mean and worst-case (max absolute) bias over all trials.

    Parameters
    ----------
    params : dict
        A dictionary containing all parameters for the simulation point:
        - 'm_main', 'm_witness': Target modulation depths.
        - 'fitter_func_name': String name of the fitting function to use.
        - 'n_trials': Number of Monte Carlo trials to run.
        - 'witness_phi_uncertainty_rad': Half-width of the uniform random
          distribution for the phase error (in radians).
        - 'grid_i', 'grid_j': Indices for the output grid.

    Returns
    -------
    tuple
        A tuple (grid_i, grid_j, mean_bias, worst_case_bias).
    """
    # I've designed this to be self-contained for robust parallel execution.
    import DeepFMKit.core as dfm
    import numpy as np

    # Unpack parameters from the input dictionary
    m_main_target = params['m_main']
    m_witness_target = params['m_witness']
    fitter_func_name = params['fitter_func_name']
    n_trials = params['n_trials']
    witness_phi_uncertainty_rad = params['witness_phi_uncertainty_rad']
    grid_i = params['grid_i']
    grid_j = params['grid_j']
    force_witness_phase = params['force_witness_phase']

    biases_for_this_point = []

    # --- Monte Carlo Loop ---
    for _ in range(n_trials):
        dff = dfm.DeepFitFramework()

        # 1. Configure laser and main IFO
        laser_config = dfm.LaserConfig()
        main_ifo_config = dfm.InterferometerConfig(label="main_ifo")
        laser_config.df_2nd_harmonic_frac = 0.0 # No distortion

        opd_main = main_ifo_config.meas_arml - main_ifo_config.ref_arml
        if opd_main == 0: continue
        laser_config.df = (m_main_target * dfm.sc.c) / (2 * np.pi * opd_main)
        
        main_label = "main"
        main_channel = dfm.DFMIObject(main_label, laser_config, main_ifo_config)
        dff.sims[main_label] = main_channel

        # 2. Design Witness IFO with Stochastic Phase
        witness_label = "witness"
        opd_witness = (m_witness_target * dfm.sc.c) / (2 * np.pi * laser_config.df)

        witness_ifo_config = dfm.InterferometerConfig(label="witness_ifo")
        witness_ifo_config.ref_arml = 0.01
        witness_ifo_config.meas_arml = witness_ifo_config.ref_arml + opd_witness

        # Calculate the ideal quadrature point
        f0 = dfm.sc.c / laser_config.wavelength
        static_fringe_phase = (2 * np.pi * f0 * opd_witness) / dfm.sc.c
        ideal_phi_offset = (np.pi / 2.0) - static_fringe_phase
        
        # Add a random error to simulate imperfect fringe locking
        lock_error = np.random.uniform(-witness_phi_uncertainty_rad, witness_phi_uncertainty_rad)
        witness_ifo_config.phi = ideal_phi_offset + lock_error

        if force_witness_phase is not None: witness_ifo_config.phi = force_witness_phase

        witness_channel = dfm.DFMIObject(witness_label, laser_config, witness_ifo_config, f_samp=main_channel.f_samp)
        dff.sims[witness_label] = witness_channel

        # 3. Simulate and Fit for this single trial
        n_seconds = main_channel.fit_n / laser_config.f_mod
        dff.simulate(main_label, n_seconds=n_seconds, witness_label=witness_label, mode='asd')
        
        try:
            fitter_function = getattr(dff, fitter_func_name)
        except AttributeError:
            biases_for_this_point.append(np.nan)
            continue

        fit_obj = fitter_function(main_label, witness_label, fit_label="wdfmi_fit", verbose=False)
        
        if fit_obj and fit_obj.m.size > 0:
            bias = fit_obj.m[0] - m_main_target
            biases_for_this_point.append(bias)
        else:
            biases_for_this_point.append(np.nan)

    # --- 4. Calculate Summary Statistics ---
    if not biases_for_this_point: # Handle case where all trials failed
        return (grid_i, grid_j, np.nan, np.nan)
        
    biases_arr = np.array(biases_for_this_point)
    mean_bias = np.nanmean(biases_arr)
    worst_case_bias = np.nanmax(np.abs(biases_arr))

    return (grid_i, grid_j, mean_bias, worst_case_bias)

def calculate_wdfmi_vs_dfmi_bias_with_distortion(params):
    """
    Worker function for a stochastic analysis comparing W-DFMI and conventional
    DFMI performance in the presence of modulation non-linearity.

    For a single point in the (m_main, distortion_amplitude) space, this function
    runs a Monte Carlo simulation. In each trial, the *phase* of the second
    harmonic distortion is randomized. It then fits the distorted signal with
    both the specified W-DFMI fitter and the conventional NLS fitter.

    This allows for a direct, quantitative comparison of how each algorithm's
    bias behaves under the exact same, challenging signal conditions.

    Parameters
    ----------
    params : dict
        A dictionary containing all parameters for the simulation point:
        - 'm_main': Target modulation depth for the main channel.
        - 'm_witness': Fixed modulation depth for the witness channel.
        - 'distortion_amp': Amplitude of the 2nd harmonic distortion (epsilon).
        - 'fitter_func_name': String name of the W-DFMI fitting function to use.
        - 'n_trials': Number of Monte Carlo trials to run.
        - 'grid_i', 'grid_j': Indices for the output grid.

    Returns
    -------
    tuple
        A tuple (grid_i, grid_j, wdfmi_mean_bias, wdfmi_worst_bias,
                 dfmi_mean_bias, dfmi_worst_bias).
    """
    # Self-contained imports for parallel processing
    import DeepFMKit.core as dfm
    import numpy as np

    # Unpack parameters
    m_main_target = params['m_main']
    m_witness_target = params['m_witness']
    distortion_amp = params['distortion_amp']
    fitter_func_name = params['fitter_func_name']
    n_trials = params['n_trials']
    grid_i = params['grid_i']
    grid_j = params['grid_j']

    wdfmi_biases = []
    dfmi_biases = []

    # --- Monte Carlo Loop over random distortion phase ---
    for _ in range(n_trials):
        dff = dfm.DeepFitFramework()

        # 1. Configure laser and main IFO
        laser_config = dfm.LaserConfig()
        main_ifo_config = dfm.InterferometerConfig(label="main_ifo")
        
        # Inject the non-linearity
        laser_config.df_2nd_harmonic_frac = distortion_amp
        laser_config.df_2nd_harmonic_phase = np.random.uniform(0, 2 * np.pi)

        # Set laser df for the target m_main
        opd_main = main_ifo_config.meas_arml - main_ifo_config.ref_arml
        if opd_main == 0: continue
        laser_config.df = (m_main_target * dfm.sc.c) / (2 * np.pi * opd_main)

        main_label = "main"
        main_channel = dfm.DFMIObject(main_label, laser_config, main_ifo_config)
        dff.sims[main_label] = main_channel

        # 2. Design the witness IFO
        witness_label = "witness"
        opd_witness = (m_witness_target * dfm.sc.c) / (2 * np.pi * laser_config.df)
        witness_ifo_config = dfm.InterferometerConfig(label="witness_ifo")
        witness_ifo_config.ref_arml = 0.01
        witness_ifo_config.meas_arml = witness_ifo_config.ref_arml + opd_witness
        
        # Lock witness to quadrature
        f0 = dfm.sc.c / laser_config.wavelength
        static_fringe_phase = (2 * np.pi * f0 * opd_witness) / dfm.sc.c
        witness_ifo_config.phi = (np.pi / 2.0) - static_fringe_phase
        
        witness_channel = dfm.DFMIObject(witness_label, laser_config, witness_ifo_config, f_samp=main_channel.f_samp)
        dff.sims[witness_label] = witness_channel

        # 3. Simulate and Fit for this trial
        n_seconds = main_channel.fit_n / laser_config.f_mod
        dff.simulate(main_label, n_seconds=n_seconds, witness_label=witness_label, mode='asd')
        
        # --- Run W-DFMI Fitter ---
        try:
            fitter_function = getattr(dff, fitter_func_name)
            fit_wdfmi = fitter_function(main_label, witness_label, fit_label="wdfmi_fit", verbose=False)
            if fit_wdfmi and fit_wdfmi.m.size > 0:
                wdfmi_biases.append(fit_wdfmi.m[0] - m_main_target)
            else:
                wdfmi_biases.append(np.nan)
        except Exception:
            wdfmi_biases.append(np.nan)

        # --- Run Conventional DFMI Fitter ---
        # The conventional fitter does not use the witness signal
        try:
            fit_dfmi = dff.fit(main_label, fit_label="dfmi_fit", verbose=False, parallel=False)
            if fit_dfmi and fit_dfmi.m.size > 0:
                dfmi_biases.append(fit_dfmi.m[0] - m_main_target)
            else:
                dfmi_biases.append(np.nan)
        except Exception:
            dfmi_biases.append(np.nan)
    
    # --- 4. Calculate Summary Statistics ---
    wdfmi_mean_bias, wdfmi_worst_bias = np.nanmean(wdfmi_biases), np.nanmax(np.abs(wdfmi_biases))
    dfmi_mean_bias, dfmi_worst_bias = np.nanmean(dfmi_biases), np.nanmax(np.abs(dfmi_biases))

    return (grid_i, grid_j, wdfmi_mean_bias, wdfmi_worst_bias, dfmi_mean_bias, dfmi_worst_bias)
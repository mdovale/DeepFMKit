import DeepFMKit.core as dfm
import numpy as np
import scipy.constants as sc


# def run_single_trial(params):
#     """
#     Worker function for a single simulation/fit trial. Executed by a pool process.

#     This function is self-contained. It creates its own framework instance to avoid
#     sharing objects across processes. It simulates one data set with a unique trial
#     number (for seeding) and returns the single float result.

#     Parameters
#     ----------
#     params : dict
#         A dictionary containing all parameters for this single trial run.

#     Returns
#     -------
#     float
#         The result of the fit. If sim_type is 'stat', this is the fitted 'm'.
#         If sim_type is 'sys', this is the bias ('m_fit' - 'm_true').
#     """
#     # Unpack parameters
#     trial_num = params['trial_num']
#     T_acq = params['T_acq']
#     sim_type = params['sim_type']
#     m_true = params['m_true']
#     delta_f = params['delta_f']
#     amp_asd = params['amp_asd']
#     freq_asd = params['freq_asd']
#     ndata = params['ndata']

#     # Each process gets its own instance of the framework
#     dff = dfm.DeepFitFramework()
#     label = f"worker_{trial_num}"
#     dff.new_sim(label)
#     sim = dff.sims[label]
#     sim.m = m_true
#     sim.df = delta_f
#     sim.ndata = ndata

#     # Configure noise based on simulation type
#     if sim_type == 'stat':
#         sim.amp_n = amp_asd
#         sim.f_n = 0.0
#     elif sim_type == 'sys':
#         sim.amp_n = 0.0
#         sim.f_n = freq_asd

#     # Simulate and fit the entire acquisition as one buffer
#     dff.simulate(label, n_seconds=T_acq, trial_num=trial_num)
#     fit_n = int(sim.f_mod * T_acq)
#     fit_obj = dff.fit(label, n=fit_n, verbose=False, parallel=False)

#     if fit_obj and fit_obj.m.size > 0:
#         m_fit = fit_obj.m[-1]
#         if sim_type == 'stat':
#             return m_fit
#         elif sim_type == 'sys':
#             return m_fit - m_true
    
#     return m_true if sim_type == 'stat' else 0.0


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
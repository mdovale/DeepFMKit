from DeepFMKit import physics

import numpy as np
import scipy.constants as sc
from abc import ABC, abstractmethod
from typing import Callable, Set

class ExperimentFactory(ABC):
    """
    Abstract Base Class for creating experiment configuration factories.

    Users should create a subclass of ExperimentFactory and implement the
    `__call__` method. This provides a robust and pickleable way to define
    the logic for generating physics configurations for each trial in an
    experiment, directly within a user script or notebook.
    """
    @abstractmethod
    def __call__(self, params: dict) -> dict:
        """
        Generates the physics configurations for a single experimental trial.

        Parameters
        ----------
        params : dict
            A dictionary containing all parameters (axis, static, and stochastic)
            for the specific trial being configured.

        Returns
        -------
        dict
            A dictionary containing the fully configured physics objects, typically
            including 'laser_config', 'main_ifo_config', and an optional
            'witness_ifo_config'.
        """
        pass

    @abstractmethod
    def _get_expected_params_keys(self) -> Set[str]:
        """
        Returns a set of strings representing the names of all top-level parameters
        (axis, static, stochastic) that this factory's `__call__` method expects
        to find in its `params` dictionary.

        This is used by the `Experiment` class for input validation.
        """
        pass


class StandardDFMIExperimentFactory(ExperimentFactory):
    """
    A generic, configurable factory for standard DFMI experiments.

    This factory is pickle-safe and designed to be instantiated by the user
    in their script or notebook, capturing all necessary configuration logic.
    """
    def __init__(self, waveform_function: Callable, opd_main: float = 0.1):
        """
        Initializes the factory, capturing the user's specific logic.

        Parameters
        ----------
        waveform_function : callable
            A function that generates the unitless modulation waveform.
        opd_main : float, optional
            The optical path difference of the main interferometer in meters.
        """
        if not callable(waveform_function):
            raise TypeError("waveform_function must be a callable.")
        self.waveform_func_to_use = waveform_function
        self.opd_main = opd_main

    def _get_expected_params_keys(self) -> Set[str]:
        """
        Declares the top-level parameters consumed by this factory's __call__ method.
        """
        # Note: 'distortion_amp' and 'distortion_phase' are now expected as direct
        # top-level parameters in the 'params' dict, even if they originate from
        # the waveform function's arguments.
        return {'m_main', 'psi', 'phi', 'distortion_amp', 'distortion_phase', 'waveform_kwargs'}


    def __call__(self, params: dict) -> dict:
        m_main = params['m_main']

        # Extract waveform-specific parameters directly from params,
        # providing default values if they are not explicitly set (e.g., in a sweep)
        distortion_amp = params.get('distortion_amp', 0.0)
        distortion_phase = params.get('distortion_phase', 0.0)

        # Construct the waveform_kwargs dictionary within the factory
        waveform_kwargs = {
            'distortion_amp': distortion_amp,
            'distortion_phase': distortion_phase
        }

        laser_config = physics.LaserConfig()
        laser_config.psi = params.get('psi', 0) # Allows 'psi' to be an optional top-level param
        
        main_ifo_config = physics.InterferometerConfig(label="main_ifo")
        main_ifo_config.ref_arml = 0.1
        main_ifo_config.meas_arml = main_ifo_config.ref_arml + self.opd_main
        main_ifo_config.phi = params.get('phi', 0) # Allows 'phi' to be an optional top-level param
        
        laser_config.waveform_func = self.waveform_func_to_use
        laser_config.waveform_kwargs = waveform_kwargs # Pass the constructed dict
        laser_config.df = (m_main * sc.c) / (2 * np.pi * self.opd_main)
        
        return {
            'laser_config': laser_config,
            'main_ifo_config': main_ifo_config
        }

class StandardWDFMIExperimentFactory(ExperimentFactory):
    """
    A generic, configurable factory for standard W-DFMI experiments.

    This factory is pickle-safe and designed to be instantiated by the user
    in their script or notebook, capturing all necessary configuration logic.
    """
    def __init__(self, waveform_function: Callable, opd_main: float = 0.2):
        """
        Initializes the factory, capturing the user's specific logic.

        Parameters
        ----------
        waveform_function : callable
            A function that generates the unitless modulation waveform.
        opd_main : float, optional
            The optical path difference of the main interferometer in meters.
        """
        if not callable(waveform_function):
            raise TypeError("waveform_function must be a callable.")
        self.waveform_func_to_use = waveform_function
        self.opd_main = opd_main

    def _get_expected_params_keys(self) -> Set[str]:
        """
        Declares the top-level parameters consumed by this factory's __call__ method.
        """
        return {'m_main', 'm_witness', 'psi', 'phi', 'distortion_amp', 'distortion_phase', 'waveform_kwargs'}

    def __call__(self, params: dict) -> dict:
        m_main = params['m_main']
        # Use .get() for m_witness as it might not be present in all experiments
        m_witness = params.get('m_witness', 0.0)
        
        # Extract waveform-specific parameters directly from params
        distortion_amp = params.get('distortion_amp', 0.0)
        distortion_phase = params.get('distortion_phase', 0.0)

        waveform_kwargs = {
            'distortion_amp': distortion_amp,
            'distortion_phase': distortion_phase
        }

        laser_config = physics.LaserConfig()
        laser_config.psi = params.get('psi', 0)

        main_ifo_config = physics.InterferometerConfig(label="main_ifo")
        main_ifo_config.ref_arml = 0.1
        main_ifo_config.meas_arml = main_ifo_config.ref_arml + self.opd_main
        main_ifo_config.phi = params.get('phi', 0)

        laser_config.waveform_func = self.waveform_func_to_use
        laser_config.waveform_kwargs = waveform_kwargs
        
        laser_config.df = (m_main * sc.c) / (2 * np.pi * self.opd_main)
        
        witness_ifo_config = physics.InterferometerConfig(label="witness_ifo")
        if laser_config.df > 0 and m_witness > 0:
            opd_witness = (m_witness * sc.c) / (2 * np.pi * laser_config.df)
            witness_ifo_config.ref_arml = 0.01
            witness_ifo_config.meas_arml = witness_ifo_config.ref_arml + opd_witness
            f0 = sc.c / laser_config.wavelength
            static_fringe_phase = (2 * np.pi * f0 * opd_witness) / sc.c
            witness_ifo_config.phi = (np.pi / 2.0) - static_fringe_phase
        
        return {
            'laser_config': laser_config,
            'main_ifo_config': main_ifo_config,
            'witness_ifo_config': witness_ifo_config
        }
    

"""
User adds custom factories below
"""
class VairableAmplitudeOffset(ExperimentFactory):
    def __init__(self, opd_main: float = 0.1):
        self.opd_main = opd_main

    def _get_expected_params_keys(self) -> Set[str]:
        """
        Declares the top-level parameters consumed by this factory's __call__ method.
        """
        return {'m_main', 'nominal_amplitude', 'amplitude_offset', 'waveform_kwargs'}


    def __call__(self, params: dict) -> dict:
        m_main = params['m_main']
        nominal_amplitude = params['nominal_amplitude']
        amplitude_offset = params['amplitude_offset']

        # --- Configure Laser ---
        laser_config = physics.LaserConfig(label="ExperimentLaser")
        laser_config.amp = nominal_amplitude + amplitude_offset 
        
        # Calculate df based on the desired m_main and fixed opd_main
        if self.opd_main == 0:
            raise ValueError("opd_main cannot be zero in the factory.")
        laser_config.df = (m_main * sc.c) / (2 * np.pi * self.opd_main)

        # --- Configure Interferometer ---
        main_ifo_config = physics.InterferometerConfig(label="main_ifo")
        main_ifo_config.ref_arml = 0.1
        main_ifo_config.meas_arml = main_ifo_config.ref_arml + self.opd_main

        return {
            'laser_config': laser_config,
            'main_ifo_config': main_ifo_config
        }
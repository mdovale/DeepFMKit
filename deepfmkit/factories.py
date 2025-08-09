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
"""Factories for generating physics configurations in high-throughput experiments.

This module provides classes that follow the Factory software pattern. The
primary purpose is to encapsulate user-defined logic for creating physics
configurations (`LaserConfig`, `IfoConfig`) in a way that is "pickleable"
and thus safe for use with Python's `multiprocessing` library.

Users should create a subclass of `ExperimentFactory` and implement the
`__call__` method to define the logic for a single experimental trial. This
provides a robust and portable way to define complex experiments.
"""

from deepfmkit import physics

import numpy as np
import scipy.constants as sc
from abc import ABC, abstractmethod
from typing import Callable, Set


class ExperimentFactory(ABC):
    """Abstract Base Class for creating experiment configuration factories.

    This ABC defines the required interface for all factory classes used by
    the `Experiment` controller. Subclassing this ensures that the user's
    custom logic can be correctly integrated into the parallel processing
    framework.
    """

    @abstractmethod
    def __call__(self, params: dict) -> dict:
        """Generates the physics configurations for a single experimental trial.

        This method is called by the `Experiment` worker for each individual
        trial. It must take a dictionary of parameters for that trial and
        return a dictionary of fully configured physics objects.

        Parameters
        ----------
        params : dict
            A dictionary containing all parameters (axis, static, and
            stochastic) for the specific trial being configured.

        Returns
        -------
        dict
            A dictionary containing the configured physics objects. It should
            include 'laser_config' and 'main_ifo_config'.
        """
        pass

    @abstractmethod
    def _get_expected_params_keys(self) -> Set[str]:
        """Returns the set of parameter keys this factory expects.

        This method must be implemented to declare all top-level parameter
        names (from static, axis, or stochastic definitions) that the `__call__`
        method uses. This is crucial for the `Experiment` class to perform
        input validation before launching a run.

        Returns
        -------
        set[str]
            A set of strings representing the names of all parameters this
            factory's `__call__` method expects to find in its `params` dict.
        """
        pass


class StandardDFMIExperimentFactory(ExperimentFactory):
    """A configurable factory for standard single-channel DFMI experiments.

    This factory generates configurations for a basic DFMI setup consisting of
    one laser and one interferometer. It allows for the use of custom
    modulation waveforms.

    Parameters
    ----------
    waveform_function : Callable
        A Python function that generates the unitless modulation waveform.
        It will be assigned to `laser_config.waveform_func`.
    opd_main : float, optional
        The fixed optical path difference of the main interferometer in meters.
        Defaults to 0.1.
    """

    def __init__(self, waveform_function: Callable, opd_main: float = 0.1):
        if not callable(waveform_function):
            raise TypeError("waveform_function must be a callable.")
        self.waveform_func_to_use = waveform_function
        self.opd_main = opd_main
        if self.opd_main == 0:
            raise ValueError("opd_main cannot be zero in the factory.")

    def _get_expected_params_keys(self) -> Set[str]:
        """Declares the top-level parameters consumed by this factory's __call__ method."""
        return {
            "m_main",
            "psi",
            "phi",
            "distortion_amp",
            "distortion_phase",
            "waveform_kwargs",
        }

    def __call__(self, params: dict) -> dict:
        """Generates the physics configurations for one standard DFMI trial."""
        m_main = params["m_main"]
        distortion_amp = params.get("distortion_amp", 0.0)
        distortion_phase = params.get("distortion_phase", 0.0)
        waveform_kwargs = {
            "distortion_amp": distortion_amp,
            "distortion_phase": distortion_phase,
        }
        laser_config = physics.LaserConfig()
        laser_config.psi = params.get("psi", 0)
        main_ifo_config = physics.IfoConfig(label="main_ifo")
        main_ifo_config.ref_arml = 0.1
        main_ifo_config.meas_arml = main_ifo_config.ref_arml + self.opd_main
        main_ifo_config.phi = params.get("phi", 0)
        laser_config.waveform_func = self.waveform_func_to_use
        laser_config.waveform_kwargs = waveform_kwargs
        laser_config.df = (m_main * sc.c) / (2 * np.pi * self.opd_main)
        return {"laser_config": laser_config, "main_ifo_config": main_ifo_config}


"""
User adds custom factories below
"""


class VairableAmplitudeOffset(ExperimentFactory):
    def __init__(self, opd_main: float = 0.1):
        self.opd_main = opd_main
        if self.opd_main == 0:
            raise ValueError("opd_main cannot be zero in the factory.")

    def _get_expected_params_keys(self) -> Set[str]:
        """Declares the top-level parameters consumed by this factory's __call__ method."""
        return {"m_main", "nominal_amplitude", "amplitude_offset"}

    def __call__(self, params: dict) -> dict:
        """Generates the physics configurations for one standard DFMI trial."""
        m_main = params["m_main"]
        nominal_amplitude = params["nominal_amplitude"]
        amplitude_offset = params["amplitude_offset"]
        laser_config = physics.LaserConfig(label="ExperimentLaser")
        laser_config.amp = nominal_amplitude + amplitude_offset
        laser_config.df = (m_main * sc.c) / (2 * np.pi * self.opd_main)
        main_ifo_config = physics.IfoConfig(label="main_ifo")
        main_ifo_config.ref_arml = 0.1
        main_ifo_config.meas_arml = main_ifo_config.ref_arml + self.opd_main
        return {"laser_config": laser_config, "main_ifo_config": main_ifo_config}
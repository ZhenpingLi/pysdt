from abc import ABC, abstractmethod

import numpy as np

# Placeholder imports for classes that OrbitIF methods might use
from algorithm.algorithm_def import AlgorithmDef
from orbit.model_time import ModelTime
from training.training_set import TrainingSet


class OrbitIF(ABC):
    """
    Abstract Base Class (ABC) defining the interface for orbit-related time modeling.
    
    Implementations of this interface provide methods to calculate and retrieve 
    various time parameters crucial for satellite telemetry data processing, 
    such as pattern periods, reference times, and session boundaries, often 
    considering orbital mechanics.
    """

    @abstractmethod
    def get_default_model_time(self, mnemonic_id: str) -> ModelTime:
        """
        Retrieves the default ModelTime object for a given mnemonic.
        
        Args:
            mnemonic_id (str): The identifier for the telemetry mnemonic.

        Returns:
            ModelTime: An object containing default time parameters for the model.
        """
        pass

    @abstractmethod
    def get_default_model_time_for_prev_session(self, mnemonic_id: str) -> ModelTime:
        """
        Retrieves the default ModelTime object for the previous training session.
        
        This is typically used for incremental training, where the previous 
        session's time context is needed.

        Args:
            mnemonic_id (str): The identifier for the telemetry mnemonic.

        Returns:
            ModelTime: An object containing time parameters from the previous session.
        """
        pass

    @abstractmethod
    def get_model_time(self, alg: AlgorithmDef, frequency: float, training_set: TrainingSet) -> ModelTime:
        """
        Calculates and returns a ModelTime object based on the algorithm, 
        data frequency, and the training set.

        Args:
            alg (AlgorithmDef): The algorithm definition.
            frequency (float): The sampling frequency of the data.
            training_set (TrainingSet): The data set used for training.

        Returns:
            ModelTime: An object containing calculated time parameters.
        """
        pass

    @abstractmethod
    def get_input_trend_times(self, pattern_period: float, session_time: float, num_pattern_in_training: int, is_orbitbased: bool) -> np.ndarray:
        """
        Calculates an array of timestamps representing the input trend times.
        
        These times define the specific points or ranges for which input 
        trends should be generated or retrieved.

        Args:
            pattern_period (float): The duration of one pattern cycle in seconds.
            session_time (float): The current session's reference time.
            num_pattern_in_training (int): The number of pattern cycles to consider.
            is_orbitbased (bool): Flag indicating if the calculation should 
                consider orbital mechanics.

        Returns:
            np.ndarray: An array of timestamps.
        """
        pass

    @abstractmethod
    def get_session_time(self) -> np.ndarray:
        """
        Retrieves an array of timestamps defining the current session's time boundaries.

        Returns:
            np.ndarray: An array of session-related timestamps.
        """
        pass

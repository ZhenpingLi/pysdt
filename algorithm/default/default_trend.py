from typing import List
import numpy as np
import sys
import os

from config.sdt_constants import DEFAULT, MEAN_INDEX

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config.sdt_config as sdt_config
from algorithm.data_trend import DataTrend


class DefaultTrend(DataTrend):
    """
    Predictive model implementation for 'default' (baseline) telemetry trends.
    
    This model represents telemetry that is generally constant over time or 
    contains random Gaussian noise. The predicted value is either a learned 
    single constant or falls back to the historical mean of the relevant 
    pattern cycle.
    """

    def __init__(self, mnemonic_id: str):
        """
        Initializes a new DefaultTrend object.

        Args:
            mnemonic_id (str): Unique identifier for the telemetry mnemonic.
        """
        super().__init__(mnemonic_id)
        self.param_dim = 1
        self.num_pattern_in_training = 1
        self.alg_name = DEFAULT

    def get_trend_value(self, time: List[float]) -> float:
        """
        Calculates the predicted trend value for a given time point.
        
        For the default model, this is the learned constant (params[0]). 
        If parameters are not yet trained, it returns the mean value of the 
        closest historical statistics record.

        Args:
            time (List[float]): A vector containing the target timestamp 
                as the first element.

        Returns:
            float: The predicted value (constant or mean).
        """
        if self.params is not None:
            return float(self.params[0])
        else:
            stat_point = self.get_stat_at_time(time[0])
            if stat_point:
                return float(stat_point.data[MEAN_INDEX])
            return 0.0

    def set_model_params(self, p: List[float], ref_time: float):
        """
        Populates the model from a flat parameter array and defines pattern cycles.
        
        It calculates a fixed 48-hour session window (split into two 24h 
        pattern cycles) relative to the reference time.

        Args:
            p (List[float]): The source parameter array.
            ref_time (float): The reference start timestamp.
        """
        super().set_model_params(p, ref_time)
        session_period = sdt_config.get_float_property("SESSIONPERIOD") * 3600
        if session_period == 0:
            session_period = 48 * 3600
            
        self.pattern_period = session_period / 2.0
        self.pattern_times = np.array([ref_time, ref_time + self.pattern_period, ref_time + session_period])

    def get_model_params(self) -> List[float]:
        """
        Serializes the model's learned state into a flat list.
        
        If parameters were not explicitly trained, it uses the first 
        historical mean as the model constant.

        Returns:
            List[float]: The serialized parameter array.
        """
        if self.params is None:
            self.params = [0.0] * self.param_dim
            if self.stat_list and self.stat_list[0]:
                self.params[0] = self.stat_list[0].data[MEAN_INDEX]
        
        return super().get_model_params()

    def get_model_param_dim(self) -> int:
        """
        Returns the total length of the serialized model representation.

        Returns:
            int: The parameter count (3 in this implementation).
        """
        return 1 + 2 # 1 for constant param + 2 for temporal context metadata

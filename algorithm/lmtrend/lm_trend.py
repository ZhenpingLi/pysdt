from typing import List
import sys
import os

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_trend import DataTrend

class LMTrend(DataTrend):
    """
    Represents a simple limit-based data trend model.
    In this model, the trend is a constant value, typically representing a mean.
    """

    def __init__(self, mnemonic_id : str):
        super().__init__(mnemonic_id)
        self.param_dim = 1

    def get_trend_value(self, time: List[float]) -> float:
        """
        Gets the trend value, which is simply the constant parameter of this model.
        """
        if self.params and len(self.params) > 0:
            return float(self.params[0])
        return 0.0

    def get_model_param_dim(self) -> int:
        """
        Gets the dimension of the model's parameters (value and sigma).
        """
        return 2

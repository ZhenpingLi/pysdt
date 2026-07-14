import os
import sys
from typing import List

from algorithm.multivariable.multi_variable_trend import MultiVariableTrend
from algorithm.multivariable.multi_variable_trend_worker import MultiVariableTrendWorker

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.algorithm_factory import AlgorithmFactory, TrainingWorker
from algorithm.data_trend import DataTrend


class MultiVariableTrendAlgorithmFactory(AlgorithmFactory):
    """
    Factory implementation for the Multi-Variable Polynomial Trend algorithm.
    
    This factory is responsible for instantiating the components required for 
    modeling telemetry that depends on multiple input dimensions (e.g., time 
    plus another telemetry point). It provides the MultiVariableTrendWorker 
    for model fitting and the MultiVariableTrend for predictions.
    """
    MULTRIVARIABLEPOLY = "mpoly"

    def get_training_worker(self) -> TrainingWorker:
        """
        Instantiates and returns the multi-variable training worker.

        Returns:
            TrainingWorker: An instance of MultiVariableTrendWorker.
        """
        return MultiVariableTrendWorker()

    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        """
        Instantiates and returns the multi-variable trend model.

        Args:
            mnemonic_id (str): The unique identifier for the telemetry mnemonic.

        Returns:
            DataTrend: An instance of MultiVariableTrend configured for 
                the mnemonic.
        """
        return MultiVariableTrend(mnemonic_id)

    def get_algorithm_name(self) -> str:
        """
        Returns the logical name of this algorithm strategy.

        Returns:
            str: The constant string 'mpoly'.
        """
        return self.MULTRIVARIABLEPOLY

from typing import List
import sys
import os

from config.sdt_constants import DEFAULT

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.algorithm_factory import AlgorithmFactory, TrainingWorker
from algorithm.default.default_trend import DefaultTrend
from algorithm.data_trend import DataTrend
from algorithm.default.default_training_worker import DefaultTrainingWorker


class DefaultAlgorithmFactory(AlgorithmFactory):
    """
    Factory implementation for the 'default' (baseline) training algorithm.
    
    This factory is responsible for instantiating the core components used 
    for modeling telemetry that is generally constant or contains only random 
    noise. It provides the DefaultTrainingWorker for fitting and the 
    DefaultTrend for prediction.
    """

    def get_training_worker(self) -> TrainingWorker:
        """
        Instantiates and returns the default training worker.

        Returns:
            TrainingWorker: An instance of DefaultTrainingWorker.
        """
        return DefaultTrainingWorker()

    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        """
        Instantiates and returns the default trend model object.

        Args:
            mnemonic_id (str): The unique identifier for the telemetry mnemonic.

        Returns:
            DataTrend: An instance of DefaultTrend configured for the mnemonic.
        """
        return DefaultTrend(mnemonic_id)

    def get_algorithm_name(self) -> str:
        """
        Returns the logical name of this algorithm strategy.

        Returns:
            str: The constant string 'default'.
        """
        return DEFAULT

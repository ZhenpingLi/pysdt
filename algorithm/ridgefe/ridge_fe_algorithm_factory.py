from typing import List
import sys
import os

from .ridge_fe_training_worker import RidgeFETrainingWorker

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.algorithm_factory import AlgorithmFactory, TrainingWorker
from algorithm.data_trend import DataTrend
from algorithm.ridgefe.ridge_fe_trend import RidgeFeTrend


class RidgeFEAlgorithmFactory(AlgorithmFactory):
    """
    Factory implementation for the Ridge Fourier Expansion (RidgeFE) algorithm.
    
    This factory is responsible for instantiating the core components used 
    for modeling periodic telemetry patterns using regularized Fourier 
    expansion. It provides the RidgeFETrainingWorker for model optimization 
    and the RidgeFeTrend for generating predictions.
    """
    RDFE = "ridgefe"

    def get_training_worker(self) -> TrainingWorker:
        """
        Instantiates and returns the RidgeFE training worker.

        Returns:
            TrainingWorker: An instance of RidgeFETrainingWorker.
        """
        return RidgeFETrainingWorker()

    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        """
        Instantiates and returns the RidgeFE trend model object.

        Args:
            mnemonic_id (str): The unique identifier for the telemetry mnemonic.

        Returns:
            DataTrend: An instance of RidgeFeTrend configured for the mnemonic.
        """
        return RidgeFeTrend(mnemonic_id)

    def get_algorithm_name(self) -> str:
        """
        Returns the logical name of this algorithm strategy.

        Returns:
            str: The constant string 'ridgefe'.
        """
        return self.RDFE

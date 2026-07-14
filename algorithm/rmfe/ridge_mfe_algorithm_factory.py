import os
import sys

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.algorithm_factory import AlgorithmFactory
from algorithm.data_trend import DataTrend
from training.training_worker import TrainingWorker
from algorithm.rmfe.ridge_mfe_training_worker import RidgeMFETrainingWorker
from algorithm.rmfe.mfe_trend import MFETrend


class RidgeMFEAlgorithmFactory(AlgorithmFactory):
    """
    Factory implementation for the Ridge Modified Fourier Expansion (RMFE).
    
    This factory is responsible for instantiating components required for 
    high-dimensional periodic modeling, specifically when telemetry depends 
    on multiple variables (e.g., time and local hour). It provides the 
    RidgeMFETrainingWorker for fitting and the MFETrend for composite 
    multi-variable predictions.
    """
    RMFE = "rmfe"

    def get_training_worker(self) -> TrainingWorker:
        """
        Instantiates and returns the RidgeMFE training worker.

        Returns:
            TrainingWorker: An instance of RidgeMFETrainingWorker.
        """
        return RidgeMFETrainingWorker()

    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        """
        Instantiates and returns the MFE trend model object.

        Args:
            mnemonic_id (str): The unique identifier for the telemetry mnemonic.

        Returns:
            DataTrend: An instance of MFETrend configured for the mnemonic.
        """
        return MFETrend(mnemonic_id)

    def get_algorithm_name(self) -> str:
        """
        Returns the logical name of this algorithm strategy.

        Returns:
            str: The constant string 'rmfe'.
        """
        return self.RMFE

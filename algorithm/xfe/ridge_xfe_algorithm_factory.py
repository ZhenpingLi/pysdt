from typing import List
import sys
import os

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.algorithm_factory import AlgorithmFactory, TrainingWorker
from algorithm.data_trend import DataTrend
from algorithm.xfe.xfe_trend import XFETrend
from algorithm.xfe.ridge_xfe_training_worker import RidgeXFETrainingWorker


class RidgeXFEAlgorithmFactory(AlgorithmFactory):
    """
    Factory implementation for the Ridge Extended Fourier Expansion (RidgeXFE) algorithm.
    
    This factory is responsible for instantiating components required for 
    high-complexity periodic modeling where Fourier series are extended with 
    additional variables. It provides the RidgeXFETrainingWorker for model 
    fitting and the XFETrend for performing predictions.
    """
    RIDGEXFE = "refea"

    def get_training_worker(self) -> TrainingWorker:
        """
        Instantiates and returns the RidgeXFE training worker.

        Returns:
            TrainingWorker: An instance of RidgeXFETrainingWorker.
        """
        return RidgeXFETrainingWorker()

    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        """
        Instantiates and returns the RidgeXFE trend model object.

        Args:
            mnemonic_id (str): The unique identifier for the telemetry mnemonic.

        Returns:
            DataTrend: An instance of XFETrend configured for the mnemonic.
        """
        return XFETrend(mnemonic_id)

    def get_algorithm_name(self) -> str:
        """
        Returns the logical name of this algorithm strategy.

        Returns:
            str: The constant string 'refea'.
        """
        return self.RIDGEXFE

from typing import List
import sys
import os

from algorithm.fbnn.fbnn_trend import FBNNTrend
from algorithm.fbnn.nn_training_worker import NNTrainingWorker
from config.sdt_constants import FBNN

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.algorithm_factory import AlgorithmFactory, TrainingWorker
from algorithm.data_trend import DataTrend


class NeuralNetAlgorithmFactory(AlgorithmFactory):
    """
    Factory implementation for the Feed-forward Back-propagation Neural Network (FBNN).
    
    This factory is responsible for instantiating the components required for 
    high-complexity telemetry modeling. It provides the NNTrainingWorker for 
    L-BFGS optimized model fitting and the FBNNTrend for performing predictions 
    based on the trained weights.
    """

    def get_training_worker(self) -> TrainingWorker:
        """
        Instantiates and returns the neural network training worker.

        Returns:
            TrainingWorker: An instance of NNTrainingWorker.
        """
        return NNTrainingWorker()

    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        """
        Instantiates and returns the FBNN trend model object.

        Args:
            mnemonic_id (str): The unique identifier for the telemetry mnemonic.

        Returns:
            DataTrend: An instance of FBNNTrend configured for the mnemonic.
        """
        return FBNNTrend(mnemonic_id)

    def get_algorithm_name(self) -> str:
        """
        Returns the logical name of this algorithm strategy.

        Returns:
            str: The constant string 'fbnn'.
        """
        return FBNN

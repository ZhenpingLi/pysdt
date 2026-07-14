from typing import List
import sys
import os

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.algorithm_factory import AlgorithmFactory, TrainingWorker
from algorithm.data_trend import DataTrend
from algorithm.lnn.lnn_trend import LNNTrend
from algorithm.lnn.lnn_training_worker import LNNTrainingWorker

class LNNAlgorithmFactory(AlgorithmFactory):
    """
    Factory for creating instances of the Liquid Neural Network (LNN) algorithm components.
    """
    LNN = "lnn"

    def get_training_worker(self) -> TrainingWorker:
        """
        Creates a new instance of LNNTrainingWorker.
        """
        return LNNTrainingWorker()

    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        """
        Creates a new instance of LNNTrend.
        """
        return LNNTrend(mnemonic_id)

    def get_algorithm_name(self) -> str:
        """
        Returns the name of the algorithm.
        """
        return self.LNN

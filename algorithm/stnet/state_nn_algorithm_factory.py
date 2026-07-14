from typing import List
import sys
import os
import numpy as np

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.algorithm_factory import AlgorithmFactory, TrainingWorker
from algorithm.data_trend import DataTrend
from algorithm.stnet.state_nn_trend import StateNNTrend
from algorithm.stnet.state_nn_training_worker import StateNNTrainingWorker


class StateNNAlgorithmFactory(AlgorithmFactory):
    """
    Factory implementation for the State-based Neural Network (STNET) algorithm.
    
    This factory is responsible for instantiating the components required for 
    modeling telemetry during specific operational states (e.g., maneuvers, 
    eclipses) using neural networks. It provides the StateNNTrainingWorker 
    for state-specific model fitting and the StateNNTrend for predictions.
    """
    STNET = "stnet"

    def get_training_worker(self) -> TrainingWorker:
        """
        Instantiates and returns the state-based neural network training worker.

        Returns:
            TrainingWorker: An instance of StateNNTrainingWorker.
        """
        return StateNNTrainingWorker()

    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        """
        Instantiates and returns the state-based neural network trend model.

        Args:
            mnemonic_id (str): The unique identifier for the telemetry mnemonic.

        Returns:
            DataTrend: An instance of StateNNTrend configured for the mnemonic.
        """
        return StateNNTrend(mnemonic_id)

    def get_algorithm_name(self) -> str:
        """
        Returns the logical name of this algorithm strategy.

        Returns:
            str: The constant string 'stnet'.
        """
        return self.STNET

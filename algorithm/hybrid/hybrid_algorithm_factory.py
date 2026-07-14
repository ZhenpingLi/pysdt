import os
import sys

from config.sdt_constants import HYBRID

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.algorithm_factory import AlgorithmFactory, TrainingWorker
from algorithm.data_trend import DataTrend
from algorithm.hybrid.hybrid_trend import HybridTrend
from algorithm.hybrid.hybrid_training_worker import HybridTrainingWorker


class HybridAlgorithmFactory(AlgorithmFactory):
    """
    Factory implementation for the 'hybrid' (multi-state) training algorithm.
    
    This factory is responsible for instantiating components required for 
    modeling telemetry using finite state machines. It provides the 
    HybridTrainingWorker for state-based model coordination and the HybridTrend 
    for managing composite multi-model predictions.
    """

    def get_training_worker(self) -> TrainingWorker:
        """
        Instantiates and returns the hybrid training worker.

        Returns:
            TrainingWorker: An instance of HybridTrainingWorker.
        """
        return HybridTrainingWorker()

    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        """
        Instantiates and returns the composite hybrid trend model.

        Args:
            mnemonic_id (str): The unique identifier for the telemetry mnemonic.

        Returns:
            DataTrend: An instance of HybridTrend configured for the mnemonic.
        """
        return HybridTrend(mnemonic_id)

    def get_algorithm_name(self) -> str:
        """
        Returns the logical name of this algorithm strategy.

        Returns:
            str: The constant string 'hybrid'.
        """
        return HYBRID

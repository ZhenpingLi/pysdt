# Add parent directory to path to find other modules
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.algorithm_factory import AlgorithmFactory, TrainingWorker
from algorithm.default.default_trend import DefaultTrend
from algorithm.data_trend import DataTrend
from algorithm.discrete.discrete_training_worker import DiscreteTrainingWorker


class DiscreteAlgorithmFactory(AlgorithmFactory):
    """
    Factory implementation for the 'discrete' training algorithm.
    
    This factory is responsible for instantiating components used for modeling 
    quantized or discrete-valued telemetry. It provides the DiscreteTrainingWorker 
    for specialized fitting and uses the DefaultTrend for prediction.
    """
    DISCRETE = "discrete"

    def get_training_worker(self) -> TrainingWorker:
        """
        Instantiates and returns the discrete training worker.

        Returns:
            TrainingWorker: An instance of DiscreteTrainingWorker.
        """
        return DiscreteTrainingWorker()

    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        """
        Instantiates and returns the trend model object.
        
        Note: The discrete algorithm shares the DefaultTrend model for prediction.

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
            str: The constant string 'discrete'.
        """
        return self.DISCRETE

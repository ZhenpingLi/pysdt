from abc import ABC, abstractmethod
from typing import List, Optional

from training.training_worker import TrainingWorker
# Assuming these classes will be defined elsewhere or mocked for now
# In a real conversion, you would import the Python equivalents of these classes.
from algorithm.data_trend import DataTrend


class AlgorithmFactory(ABC):
    """
    AIMS Algorithm Factory Interface. It is used to generate the input/output,
    data trend, and trending worker object that implements the specific time
    dependent trend algorithms.
    """

    CONTEXT = "DataTraining"
    TRAININGTYPES = ["New Training", "Retraining", "Full Training", "Full Retraining"]

    @abstractmethod
    def get_training_worker(self) -> TrainingWorker:
        """
        Returns the implementation for the data training algorithms
        :return: TrainingWorker object.
        """
        pass

    @abstractmethod
    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        """
        Create a DataTrend object.

        :param mnemonic_id: The mnemonic ID.

        :return: DataTrend Object
        """
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """
        Returns the algorithm name for the factory.

        :return: The String name for algorithm.
        """
        pass

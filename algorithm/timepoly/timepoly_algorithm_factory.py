from algorithm.algorithm_factory import AlgorithmFactory, TrainingWorker
from algorithm.data_trend import DataTrend
from typing import List

from algorithm.timepoly.time_poly_trend import TimePolyTrend
from algorithm.timepoly.time_poly_trend_worker import TimePolyTrendWorker


class TimePolyAlgorithmFactory(AlgorithmFactory):
    """
    Factory implementation for the Time-based Polynomial Trend algorithm.
    
    This factory is responsible for instantiating components required for 
    modeling telemetry trends as an explicit polynomial function of absolute 
    time. It provides the TimePolyTrendWorker for least-squares fitting and 
    the TimePolyTrend for performing predictions.
    """

    def get_training_worker(self) -> TrainingWorker:
        """
        Instantiates and returns the TimePoly training worker.

        Returns:
            TrainingWorker: An instance of TimePolyTrendWorker.
        """
        return TimePolyTrendWorker()

    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        """
        Instantiates and returns the TimePoly trend model object.

        Args:
            mnemonic_id (str): The unique identifier for the telemetry mnemonic.

        Returns:
            DataTrend: An instance of TimePolyTrend configured for the mnemonic.
        """
        return TimePolyTrend(mnemonic_id)

    def get_algorithm_name(self) -> str:
        """
        Returns the logical name of this algorithm strategy.

        Returns:
            str: The constant string 'timepoly'.
        """
        return "timepoly"

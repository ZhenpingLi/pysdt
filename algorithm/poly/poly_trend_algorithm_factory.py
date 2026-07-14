from typing import List
import sys
import os

from config.sdt_constants import POLYTREND

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.algorithm_factory import AlgorithmFactory, TrainingWorker
from algorithm.data_trend import DataTrend
from algorithm.poly.poly_trend import PolyTrend
from algorithm.poly.poly_trend_worker import PolyTrendWorker


class PolyTrendAlgorithmFactory(AlgorithmFactory):
    """
    Factory implementation for the Polynomial Trend algorithm.
    
    This factory is responsible for instantiating the components required for 
    modeling telemetry that exhibits linear or non-linear drift. It provides 
    the PolyTrendWorker for model fitting and the PolyTrend for generating 
    predictions.
    """

    def get_training_worker(self) -> TrainingWorker:
        """
        Instantiates and returns the polynomial training worker.

        Returns:
            TrainingWorker: An instance of PolyTrendWorker.
        """
        return PolyTrendWorker()

    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        """
        Instantiates and returns the polynomial trend model object.

        Args:
            mnemonic_id (str): The unique identifier for the telemetry mnemonic.

        Returns:
            DataTrend: An instance of PolyTrend configured for the mnemonic.
        """
        return PolyTrend(mnemonic_id)

    def get_algorithm_name(self) -> str:
        """
        Returns the logical name of this algorithm strategy.

        Returns:
            str: The constant string 'polytrend'.
        """
        return POLYTREND

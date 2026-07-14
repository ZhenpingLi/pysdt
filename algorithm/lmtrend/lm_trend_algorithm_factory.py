import os
import sys

from algorithm.algorithm_factory import AlgorithmFactory
from algorithm.data_trend import DataTrend
from algorithm.lmtrend.lm_trend import LMTrend
from algorithm.lmtrend.lm_trend_worker import LMTrendWorker

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class LMTrendAlgorithmFactory(AlgorithmFactory):
    """
        Algorithm factory for the LM Trend
    """
    LMTREND = "lmtrend"

    def get_algorithm_name(self) -> str:
        return LMTrendAlgorithmFactory.LMTREND

    def get_training_worker(self):
        return LMTrendWorker()

    def get_data_trend(self, mnemonic_id: str) -> DataTrend:
        return LMTrend(mnemonic_id)
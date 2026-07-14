import os
import sys
from typing import List, Optional, Dict

import plugin_manager
from algorithm.algorithm_data import AlgorithmData
from algorithm.data_point import DataPoint
from algorithm.training_output import TrainingOutputData
from algorithm.trend_node import TrendNode
from posttraining.clustering.sdt_event_data import SDTEventData
from training import data_buffer

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.algorithm_def import SHORTTERM
from algorithm.data_trend import DataTrend
from algorithm import training_output_processor as top
from algorithm.outlier import Outlier


class MnemonicNode(TrendNode):
    """
    Leaf node in the training tree representing a single telemetry mnemonic.
    
    This class manages the trained data models, detected outliers, and 
    pre-processed event data for a specific telemetry point. It handles 
    the conversion between live model objects and serialized output data.
    """
    ATTRIBUTE_KEY: List[str] = ['name', 'data_trend', 'input_trend', 'training_error', 'ops_status']

    def __init__(self, mnemonic_name: str):
        """
        Initializes a new instance of the MnemonicNode.

        Args:
            mnemonic_name (str): The name of the mnemonic (e.g., 'TEMP_1').
        """
        super().__init__(mnemonic_name)
        self.data_trend: Optional[List[DataTrend]] = None
        self.input_trend: Optional[DataTrend] = None
        self.outlier_list: Optional[List[Outlier]] = None
        self.event_data_list: Optional[List[SDTEventData]] = None
        self.sigma_mean : Dict[str, float] = None
        self.training_error: Optional[str] = None
        self.ops_status : Optional[List[DataPoint]] = None

    def set_input(self, input_trend: DataTrend):
        """
        Sets the reference (baseline) trend for the current training session.

        Args:
            input_trend (DataTrend): The baseline model.
        """
        if self.is_leaf:
            self.input_trend = input_trend

    def get_outliers(self, outlier_list: List[Outlier]):
        """
        Appends the detected outliers for this mnemonic to the provided list.

        Args:
            outlier_list (List[Outlier]): The list to which outliers will be added.
        """
        if self.outlier_list:
            outlier_list.extend(self.outlier_list)

    def add_outliers(self, outlier_list: List[Outlier]):
        """
        Adds new outliers to the node and triggers the creation of event segments.

        Args:
            outlier_list (List[Outlier]): The list of newly detected outliers.
        """
        if self.outlier_list:
            self.outlier_list.extend(outlier_list)
        else:
            self.outlier_list = outlier_list
        self.event_data_list = top.create_mnemonic_event_data_list_from_outlier_list(self.outlier_list)

    def get_event_data_list(self) -> List[SDTEventData]:
        """Returns the list of processed event data segments."""
        return self.event_data_list

    def set_training_error(self, error: str):
        """Sets an error message indicating that training failed."""
        self.training_error = error

    def is_training_defined(self) -> bool:
        """
        Determines if training should be performed for this node.
        
        Always returns True for short-term sessions. For long-term sessions, 
        only returns True if the node is a leaf.
        """
        if data_buffer.session_type == SHORTTERM :
            return True
        else:
            training_defined : bool = False
            if self.is_leaf :
                level = self.get_node_level()
                training_defined = level == 2
        return training_defined

    def get_data_trends(self) -> Optional[List[DataTrend]]:
        """Returns the list of trained DataTrend models."""
        return self.data_trend

    def get_training_output(self)-> TrainingOutputData:
        """
        Generates a TrainingOutputData object containing the current session's 
        results for serialization and storage.

        Returns:
            Optional[TrainingOutputData]: The processed output data, or None if 
                no trends exist.
        """
        data_trend_data_list : List[AlgorithmData]=[]
        if not self.data_trend:
            return None
        for trend in self.data_trend:
            algorithm_data = trend.get_algorithm_data()
            if algorithm_data is not None:
                data_trend_data_list.append(trend.get_algorithm_data())
        return TrainingOutputData(
            mnemonic_id=self.name,
            outlier_list=self.outlier_list,
            mnemonic_event_list=self.event_data_list,
            algorithm_data_list=data_trend_data_list,
            training_error=self.training_error if self.training_error else None
        )

    def set_training_output(self, training_output: TrainingOutputData):
        """
        Populates the node properties from a TrainingOutputData object.
        Reconstructs the DataTrend objects using the appropriate algorithm factory.

        Args:
            training_output (TrainingOutputData): The source output data.
        """
        self.outlier_list = training_output.outlier_list
        self.event_data_list = training_output.mnemonic_event_list
        self.training_error = training_output.training_error
        if training_output.algorithm_data_list:
            alg_name = training_output.algorithm_data_list[0].alg_name
            alg_factory = plugin_manager.get_algorithm_factory(alg_name)
            self.data_trend = []
            for algorithm_data in training_output.algorithm_data_list:
                _trend = alg_factory.get_data_trend(self.name)
                _trend.set_algorithm_data(algorithm_data)
                self.data_trend.append(_trend)

    def __getitem__(self, item):
        if item in MnemonicNode.ATTRIBUTE_KEY:
            return getattr(self, item)
        else:
            raise KeyError(f"'MnemonicNode' object has no attribute '{item}'")

    def __setitem__(self, key, value):
        if key in MnemonicNode.ATTRIBUTE_KEY:
            setattr(self, key, value)
        else:
            raise KeyError(f"'MnemonicNode' object has no attribute '{key}'")

    def __contains__(self, item):
        return item in MnemonicNode.ATTRIBUTE_KEY

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from algorithm.algorithm_data import AlgorithmData
from algorithm.data_point import DataPoint
from algorithm.outlier import Outlier
from posttraining.clustering.sdt_event_data import SDTEventData

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class TrainingOutputData:
    """
    Data structure representing the results of a data training session for a single mnemonic.
    
    This object encapsulates the learned model parameters, detected outliers, and 
    pre-processed event data, serving as the primary output for storage and 
    downstream analysis.
    
    Attributes:
        mnemonic_id (str): The unique identifier for the telemetry mnemonic.
        outlier_list (Optional[List[Outlier]]): A list of detected outliers.
        mnemonic_event_list (Optional[List[SDTEventData]]): Processed outlier events.
        algorithm_data_list (List[AlgorithmData]): List of trained models (parameters) 
            for each operational state.
        ops_status (Optional[List[DataPoint]]): Data points representing the operational 
            health metrics.
        training_error (Optional[str]): Error message if training failed for this mnemonic.
    """
    mnemonic_id : str
    outlier_list: Optional[List[Outlier]] = field(default_factory=list)
    mnemonic_event_list: Optional[List[SDTEventData]] = field(default_factory=list)
    algorithm_data_list: List[AlgorithmData] = field(default_factory=list)
    ops_status: Optional[List[DataPoint]] = None
    training_error: Optional[str] = None

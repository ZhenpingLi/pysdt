from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np

from training.training_set import TrainingSet


def pad_function(training_set : TrainingSet):
    """
    Pads the outputs of a TrainingSet based on a specific value (10.25).
    
    If an output value is 10.25, it's replaced with 1; otherwise, 0.
    This is typically used for binary padding or specific state identification.

    Args:
        training_set (TrainingSet): The training set whose outputs are to be padded.
    """
    if training_set.outputs is not None:
        pad_outputs = np.array([1 if o==10.25 else 0 for o in training_set.outputs])
        training_set.outputs = pad_outputs
        training_set.raw = pad_outputs
    return None

def norm_function(training_set : TrainingSet):
    """
    Normalizes the outputs of a TrainingSet to binary values (0 or 1).
    
    If an output value is greater than 0.01, it's replaced with 1; otherwise, 0.
    This is a simple threshold-based normalization.

    Args:
        training_set (TrainingSet): The training set whose outputs are to be normalized.
    """
    if training_set.outputs is not None:
        normalized_outputs = np.array([1 if o>0.01 else 0 for o in training_set.outputs])
        training_set.outputs = normalized_outputs
        training_set.raw = normalized_outputs
    return None

def bin_function(training_set : TrainingSet):
    """
    Converts the outputs of a TrainingSet to binary float values (0.0 or 1.0).
    
    If an output value is 1, it's replaced with 1.0; otherwise, 0.0.
    This ensures binary outputs are represented as floats.

    Args:
        training_set (TrainingSet): The training set whose outputs are to be binarized.
    """
    if training_set.outputs is not None:
        binary_outputs = np.array([1.0 if o==1 else 0.0 for o in training_set.outputs])
        training_set.outputs =binary_outputs
        training_set.raw = binary_outputs
    return None

commands = {
    "pad": pad_function,
    "norm": norm_function,
    "bin": bin_function
}
"""Dictionary mapping command strings to their respective preprocessing functions."""

def _merge_data_list(training_set: TrainingSet, column_times: List[np.ndarray], column_data: List[np.ndarray]):
    """
    Merges multiple columns of data into the input features of a TrainingSet.
    
    It interpolates the data from `column_data` onto the timestamps of the 
    `training_set.inputs` and adds them as new columns.

    Args:
        training_set (TrainingSet): The primary training set to merge data into.
        column_times (List[np.ndarray]): A list of 1D NumPy arrays, each containing 
            timestamps for a corresponding data column.
        column_data (List[np.ndarray]): A list of 1D NumPy arrays, each containing 
            data values for a corresponding data column.
    """
    time_stamps = training_set.inputs[:, 0]

    merged_matrix = np.zeros((len(time_stamps), len(column_data)+1))

    merged_matrix[:, 0] = time_stamps

    for i in range(len(column_times)):
        if column_times[i].any() and column_data[i].any():
            merged_matrix[:, i+1] = np.interp(time_stamps, column_times[i], column_data[i])

    training_set.inputs = merged_matrix

class SDTDataInput(ABC):
    """
    Abstract base class defining the contract for reading time-series data 
    from a data source (e.g., AIMS Trending Archive).
    
    Implementations of this interface provide the concrete logic for connecting 
    to specific data sources and retrieving telemetry data in a format suitable 
    for training.
    """

    def __enter__(self):
        """
        Allows the class to be used as a context manager.
        Returns:
            SDTDataInput: The instance itself.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures that the close method is called when the context is exited.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Traceback if an exception occurred.
        """
        self.close()

    @abstractmethod
    def get_data(self, mnemonic_id: str, start: float, end: float) -> Optional[TrainingSet]:
        """
        Retrieves time-tagged data points for a specified mnemonic within a time range.

        Args:
            mnemonic_id (str): The unique identifier for the telemetry mnemonic.
            start (float): The start time of the data retrieval range (Unix timestamp in seconds).
            end (float): The end time of the data retrieval range (Unix timestamp in seconds).

        Returns:
            Optional[TrainingSet]: A TrainingSet object containing inputs and outputs 
                for data training, or None if no data is found.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes any open resources, such as database connections or file handles.
        """
        pass

import os
import sys
from abc import ABC, abstractmethod
from typing import List, Optional, Any

from algorithm.algorithm_data import AlgorithmData
from posttraining.clustering.sdt_event_data import SDTEventData
from posttraining.mnemonic_status import MnemonicStatus

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.data_point import DataPoint


class SDTDataTrainingIO(ABC):
    """
    Abstract base class defining the interface for reading and writing data training results.
    
    Implementations of this interface are responsible for handling the persistence of 
    trained model parameters (AlgorithmData), statistical metrics (DataPoint), 
    and operational status events (SDTEventData) to a data archive (e.g., InfluxDB, SQLite).
    """

    def __enter__(self):
        """
        Allows the class to be used as a context manager.
        Returns:
            SDTDataTrainingIO: The instance itself.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures that resources are properly closed when the context is exited.
        """
        self.close()

    @abstractmethod
    def write_data_trend(self, algorithm_data_list: List[AlgorithmData]) -> Any:
        """
        Writes a list of trained algorithm models to the data archive.

        Args:
            algorithm_data_list (List[AlgorithmData]): The list of model data 
                objects to be persisted.

        Returns:
            Any: The result of the write operation (implementation-dependent).
        """
        pass

    @abstractmethod
    def write_status(self, status_list: List[MnemonicStatus]):
        """
        Writes the processed operational health status for multiple mnemonics 
        or subsystems to the archive.

        Args:
            status_list (List[MnemonicStatus]): The list of status objects to save.
        """
        pass

    @abstractmethod
    def get_data_trend(self, mnemonic_id: str,  start: float, end: float, state: str) -> Optional[AlgorithmData]:
        """
        Retrieves a trained model for a specific mnemonic and operational state 
        within a given time range.

        Args:
            mnemonic_id (str): The unique identifier for the telemetry mnemonic.
            start (float): The start timestamp (Unix seconds).
            end (float): The end timestamp (Unix seconds).
            state (str): The name of the operational state (e.g., 'default', 'ecl').

        Returns:
            Optional[AlgorithmData]: The retrieved model parameters, or None if not found.
        """
        pass

    @abstractmethod
    def get_data_stats(self, mnemonic_id: str, start: float, end: float, state: str) -> Optional[List[DataPoint]]:
        """
        Retrieves historical statistical metrics (mean, max, min, sigma) for a 
        given mnemonic and period.

        Args:
            mnemonic_id (str): The mnemonic identifier.
            start (float): The start timestamp.
            end (float): The end timestamp.
            state (str): The operational state.

        Returns:
            Optional[List[DataPoint]]: A list of statistical data points, or None.
        """
        pass


    @abstractmethod
    def get_sigma(self, mnemonic_id: str, start: float, end: float, state: str) -> Optional[List[DataPoint]]:
        """
        Retrieves the history of standard deviation values (sigma) for analysis 
        of temporal change.

        Args:
            mnemonic_id (str): The mnemonic identifier.
            start (float): The start timestamp.
            end (float): The end timestamp.
            state (str): The operational state.

        Returns:
            Optional[List[DataPoint]]: A list of sigma data points, or None.
        """
        pass

    @abstractmethod
    def get_event_history(self, start_time: float, end_time: float) -> Optional[List[SDTEventData]]:
        """
        Retrieves a list of outlier events that occurred within the specified 
        time range.

        Args:
            start_time (float): The start timestamp.
            end_time (float): The end timestamp.

        Returns:
            Optional[List[SDTEventData]]: The list of event data objects, or None.
        """
        pass

    @abstractmethod
    def write_events(self, event_list: List[SDTEventData], cutoff: float):
        """
        Persists a list of outlier event objects to the archive.

        Args:
            event_list (List[SDTEventData]): The events to save.
            cutoff (float): A timestamp cutoff; only events after this time 
                should be processed.
        """
        pass
        
    @abstractmethod
    def close(self):
        """
        Closes any active database connections or file handles and releases resources.
        """
        pass

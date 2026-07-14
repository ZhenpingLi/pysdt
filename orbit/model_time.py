import numpy as np
from typing import List, Optional
import sys
import os

from config.sdt_constants import DEFAULT


# Add parent directory to path to find sdtdb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdtdb import sdt_db

class ModelTime:
    """
    Data structure representing the temporal context of an analysis model.
    
    This class encapsulates session boundaries (start, end), pattern cycles 
    (period, pattern_times), and the reference timestamp. It is used to 
    coordinate time-based calculations across algorithms and training workers.
    """

    def __init__(self, start_time: float, end_time: float, model_time_id: str = DEFAULT):
        """
        Initializes the ModelTime object with session boundaries.

        Args:
            start_time (float): The absolute timestamp for the session start (Unix seconds).
            end_time (float): The absolute timestamp for the session end (Unix seconds).
            model_time_id (str): Identifier for the model context. Defaults to 'DEFAULT'.
        """
        self.start: float = start_time
        self.end: float = end_time
        self.model_time_id: str = model_time_id
        
        self.pattern_times: Optional[np.ndarray] = None
        self.ref_time: float = 0.0
        self.model_period: float = 0.0

    def set_pattern_times(self, r: np.ndarray):
        """
        Sets the starting timestamps for each pattern cycle.

        Args:
            r (np.ndarray): An array of timestamps.
        """
        self.pattern_times = r

    def get_pattern_times(self) -> Optional[np.ndarray]:
        """Returns the array of pattern start timestamps."""
        return self.pattern_times

    def set_model_period(self, p: float):
        """
        Sets the duration of a single pattern cycle.

        Args:
            p (float): The duration in seconds.
        """
        self.model_period = p

    def get_model_period(self) -> float:
        """Returns the model's pattern period in seconds."""
        return self.model_period

    def get_reference_time(self) -> float:
        """Returns the model's reference timestamp."""
        return self.ref_time

    def set_reference_time(self, r_t: float):
        """
        Sets the model's reference timestamp.

        Args:
            r_t (float): The reference time in Unix seconds.
        """
        self.ref_time = r_t

    def get_session_start(self) -> float:
        """Returns the absolute session start timestamp."""
        return self.start

    def get_session_end(self) -> float:
        """Returns the absolute session end timestamp."""
        return self.end

    def get_model_time_id(self) -> str:
        """Returns the model time identifier."""
        return self.model_time_id

    def match_orbit(self, _start: float, end: float) -> bool:
        """
        Checks if the provided start and end times correspond to this model's session.

        Args:
            _start (float): Start timestamp to check.
            end (float): End timestamp to check.

        Returns:
            bool: True if the timestamps match (within numerical precision).
        """
        return np.isclose(self.start, _start)

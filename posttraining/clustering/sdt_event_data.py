from dataclasses import dataclass

from config.sdt_constants import DEFAULT
from util import time_util


@dataclass
class SDTEventData:
    """
    Data structure representing a flat event segment derived from outliers.
    
    This dataclass holds the basic parameters for an event, including its name, 
    duration, value (intensity), and type. It serves as the building block for 
    hierarchical SDTEvent objects.

    Attributes:
        name (str): The name of the mnemonic or source.
        event_value (float): The calculated intensity or value of the event 
            (e.g., TPC ratio or cumulative outlier difference).
        event_start (float): The start timestamp in Unix seconds.
        event_end (float): The end timestamp in Unix seconds.
        event_type (str): The category of the event (e.g., 'outlier', 'tpc').
        event_state (str): The operational state during which the event occurred.
    """
    name : str
    event_value : float
    event_start : float
    event_end: float
    event_type: str
    event_state: str = DEFAULT

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the event data.
        
        Returns:
            str: A formatted string containing event details.
        """
        if self.event_type == "tpc":
            return f"{self.name}: tpc={self.event_value}\n"
        else:
            start_tag = time_util.get_time_tag_from_seconds(self.event_start)
            end_tag = time_util.get_time_tag_from_seconds(self.event_end)
            return f"{self.name}: {start_tag}-{end_tag} oc_value={self.event_value}\n"

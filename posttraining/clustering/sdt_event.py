import os
import sys
from math import sqrt
from typing import List, Optional

from config.sdt_constants import UN_DETERMINED
from posttraining.clustering.sdt_event_data import SDTEventData
from util import time_util

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SDTEvent:
    """
    Represents a hierarchical event in satellite telemetry operations.
    
    This class can represent an event at three levels: SATELLITE (top), 
    SUBSYSTEM (middle), or MNEMONIC (leaf). It supports parent-child 
    relationships to model how mnemonic-level anomalies aggregate into 
    subsystem and satellite-wide events.
    """
    SATELLITE: int = 0
    SUBSYSTEM: int = 1
    MNEMONIC: int = 2

    EVENT_KEYS: List[str] = ['name', 'event_start', 'event_end', 'event_value', 'event_type', 'event_state', 'event_level', 'event_status']

    def __init__(self, event_data: SDTEventData, event_level: int, parent: Optional['SDTEvent'] = None):
        """
        Initializes a new SDTEvent instance.

        Args:
            event_data (SDTEventData): The raw data for the event.
            event_level (int): The level (SATELLITE, SUBSYSTEM, or MNEMONIC).
            parent (Optional[SDTEvent]): The parent node in the hierarchy.
        """
        if event_data:
            self.name: str = event_data.name
            self.event_start: float = event_data.event_start
            self.event_end: float = event_data.event_end
            self.event_value: float = event_data.event_value
            self.event_type: str = event_data.event_type
            self.event_state : str = event_data.event_state
            self.event_level = event_level
            self.children : List[SDTEvent] = []
            self.event_status = UN_DETERMINED
            self.parent: Optional['SDTEvent'] = parent
            if parent is not None:
                parent.get_children().append(self)
        else:
            self.event_level = SDTEvent.MNEMONIC
            self.name: str = None
            self.event_start: float = 0.0
            self.event_end: float = 0.0
            self.event_value: float = 0.0
            self.event_type: str = None
            self.event_state: str = None
            self.parent = None
            self.children = []

    def get_child(self, child_name: str) -> Optional['SDTEvent']:
        """
        Searches for a direct child by its name.

        Args:
            child_name (str): The name to look for.

        Returns:
            Optional[SDTEvent]: The child object if found, otherwise None.
        """
        for child in self.children:
            if child['name'] == child_name:
                return child
        return None

    def get_parent(self) -> Optional['SDTEvent']:
        """Returns the parent node in the event hierarchy."""
        return self.parent

    def get_children(self) -> List['SDTEvent']:
        """Returns the list of child nodes."""
        return self.children

    def get_aggregated_event_value(self) -> float:
        """
        Calculates the aggregate event value by recursively combining child values.
        
        The aggregation uses the Euclidean norm (root-sum-of-squares) of child 
        event values.

        Returns:
            float: The aggregated event value.
        """
        if self.children and self.event_value == 0:
            aggregated_values = 0
            for child in self.children:
                child_value = child.get_aggregated_event_value()
                aggregated_values += child_value * child_value
            self.event_value = sqrt(aggregated_values)
        return self.event_value

    def __getitem__(self, item):
        """Allows attribute access via square bracket notation."""
        if item in SDTEvent.EVENT_KEYS:
            return getattr(self, item)
        else:
            raise KeyError(f"Invalid key: {item} for SDTEvent class")

    def __setitem__(self, key, value):
        """Allows attribute setting via square bracket notation."""
        if key in SDTEvent.EVENT_KEYS:
            setattr(self, key, value)
        else:
            raise KeyError(f"Invalid key: {key} for SDTEvent class")

    def __str__(self) -> str:
        """
        Generates a human-readable, indented string representation of the 
        event hierarchy.
        """
        lines = []
        if self.event_level == self.SATELLITE:
            start = time_util.get_time_tag_from_seconds(self.event_start)
            end = time_util.get_time_tag_from_seconds(self.event_end)
            lines.append(f"event start: {start}, event end: {end}:\n")
            lines.extend(str(child) for child in self.children)

        elif self.event_level == self.SUBSYSTEM:
            lines.append(f"   subsystem: {self.name}\n")
            lines.extend(str(child) for child in self.children)

        elif self.event_level == self.MNEMONIC:
            lines.append(f"      {self.name}: {self.event_value}\n")

        return "".join(lines)

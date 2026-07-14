import os
import sys
from datetime import datetime, timezone
from typing import List

from config.sdt_constants import CLUSTER
from posttraining.clustering.sdt_event import SDTEvent

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class EventCluster:
    """
    A container representing a collection of grouped satellite events.
    
    This class is used to store and manage the output of the event clustering 
    algorithm (e.g., DBSCAN). It allows for the categorization of clusters 
    (e.g., NOISE, CLUSTER) and provides utility methods for temporal analysis.
    """
    
    def __init__(self, cluster_type: int = CLUSTER):
        """
        Initializes a new EventCluster.

        Args:
            cluster_type (int): The type of the cluster (defaults to CLUSTER).
        """
        self.sdt_event_list : List[SDTEvent] = []
        self.cluster_type: int = cluster_type

    def set_cluster_type(self, cluster_type: int):
        """
        Sets the category for this cluster.

        Args:
            cluster_type (int): The new cluster type code.
        """
        self.cluster_type = cluster_type

    def add_point(self, point: SDTEvent):
        """
        Adds a hierarchical SDTEvent to the cluster.

        Args:
            point (SDTEvent): The event to add.
        """
        self.sdt_event_list.append(point)

    def get_cluster_type(self) -> int:
        """Returns the cluster type identifier."""
        return self.cluster_type

    def get_points(self) -> List[SDTEvent]:
        """Returns the list of all SDTEvent objects in this cluster."""
        return self.sdt_event_list

    def is_empty(self) -> bool:
        """Checks if the cluster contains any events."""
        return len(self.sdt_event_list) == 0

    def get_day_event_number(self) -> int:
        """
        Calculates the number of distinct calendar days spanned by the 
        events in this cluster.

        Returns:
            int: The count of unique days.
        """
        if not self.sdt_event_list:
            return 0
            
        unique_days = set()
        for event in self.sdt_event_list:
            # Normalize timestamp to date
            dt_object = datetime.fromtimestamp(event.event_start, tz=timezone.utc)
            unique_days.add(dt_object.date())
            
        return len(unique_days)

    def __str__(self) -> str:
        """
        Generates a human-readable summary of the cluster, including 
        the string representation of every contained event.
        """
        header = f"Cluster: type={self.cluster_type}, size={len(self.sdt_event_list)}\n"
        
        details = []
        for event in self.sdt_event_list:
            event_str = str(event)
            if event_str:
                details.append(event_str)
        return header + "".join(details)

import os
import sys
from typing import List, Collection, Optional

from posttraining.clustering import event_config
from posttraining.clustering.sdt_event import SDTEvent
from posttraining.clustering.event_cluster import EventCluster
from config.sdt_constants import CLUSTER, NOISE, UN_DETERMINED

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# --- Constants ---
CONTEXT = "DBSCANCLUSTER"
MAXITERATION = 100
MEANTHRESHOLD = 1.0e-5
MNEMONIC = 0

def get_event_scalar_product(event1: SDTEvent, event2: SDTEvent) -> float:
    """
    Calculates the similarity between two hierarchical events using a normalized 
    scalar product of their mnemonic-level values.
    
    The calculation only considers overlapping mnemonics and subsystems between 
    the two events.

    Args:
        event1 (SDTEvent): The first hierarchical event.
        event2 (SDTEvent): The second hierarchical event.

    Returns:
        float: The normalized scalar product (cosine similarity-like) between 
            the two events.
    """
    product = 0.0
    children1 = event1.get_children()
    for child1 in children1:
        child2 = event2.get_child(child1['name'])
        if child2 is not None:
            for grand_child1 in child1.get_children():
                grand_child2 = child2.get_child(grand_child1['name'])
                if grand_child2 is not None:
                    product += grand_child2['event_value'] * grand_child1['event_value']
    
    val1 = event1.get_aggregated_event_value()
    val2 = event2.get_aggregated_event_value()
    
    if val1 > 0 and val2 > 0:
        return float(product / (val1 * val2))
    return 0.0


def merge(one: List[SDTEvent], two: List[SDTEvent]) -> List[SDTEvent]:
    """
    Merges two lists of SDTEvents while ensuring uniqueness.

    Args:
        one (List[SDTEvent]): The primary list (modified in-place).
        two (List[SDTEvent]): The list of items to be merged into the first.

    Returns:
        List[SDTEvent]: The combined list containing unique events.
    """
    one_set = set(one)
    for item in two:
        if item not in one_set:
            one.append(item)
            one_set.add(item)
    return one


class HCluster:
    """
    Implements a density-based spatial clustering algorithm (DBSCAN) for 
    hierarchical satellite events.
    
    This class identifies groups of related anomalies by analyzing the 
    similarity (scalar product) between event trees. It categorizes events 
    into clusters or marks them as noise based on temporal and structural 
    density.
    """

    def __init__(self, cutoff_time: float):
        """
        Initializes the HCluster engine.

        Args:
            cutoff_time (float): The start timestamp for the analysis session.
        """
        self.cutoff_time = cutoff_time
        self.eps: float = event_config.eps
        self.min_pts: int = event_config.min_pts or 3
        self.debug: bool = True

    def clustering(self, events: List[SDTEvent]) -> List[EventCluster]:
        """
        Executes the clustering process on a provided list of hierarchical events.

        Args:
            events (List[SDTEvent]): The events to be analyzed.

        Returns:
            List[EventCluster]: A list of resulting clusters, including a noise cluster.
        """
        return self.db_scan_clustering(events)

    def db_scan_clustering(self, events: List[SDTEvent]) -> List[EventCluster]:
        """
        Core DBSCAN implementation for SDTEvents.
        
        Iterates through all events, identifying seed points for clusters and 
        expanding them based on the density reachability defined by 'eps' 
        and 'min_pts'.

        Args:
            events (List[SDTEvent]): The full set of events to process.

        Returns:
            List[EventCluster]: The identified clusters and the noise group.
        """
        noise_cluster = EventCluster(NOISE)
        clusters: List[EventCluster] = []
        
        # Ensure initial status is undetermined for a fresh run
        for e in events:
            e['event_status'] = UN_DETERMINED
            
        for sdt_event in events:
            if sdt_event['event_status'] == UN_DETERMINED:
                neighbors = self.get_neighbors(sdt_event, events)
                if len(neighbors) >= self.min_pts:
                    # New cluster found
                    cluster = EventCluster(CLUSTER)
                    clusters.append(self.expand_cluster(cluster, sdt_event, neighbors, events))
                else:
                    # Initially mark as noise
                    sdt_event['event_status'] = NOISE
                    noise_cluster.add_point(sdt_event)

        if not noise_cluster.is_empty():
            clusters.append(noise_cluster)

        return clusters

    def get_neighbors(self, sdt_event: SDTEvent, points: List[SDTEvent]) -> List[SDTEvent]:
        """
        Identifies all density-reachable neighbors for a given event.
        
        Neighbors are defined as events whose normalized scalar product with the 
        target event is greater than or equal to the 'eps' similarity threshold.

        Args:
            sdt_event (SDTEvent): The target event.
            points (List[SDTEvent]): The set of potential neighbors to search.

        Returns:
            List[SDTEvent]: A list of qualifying neighbor events.
        """
        neighbors = []
        for _event in points:
            if _event is not sdt_event:
                dist = get_event_scalar_product(_event, sdt_event)
                if dist >= self.eps:
                    neighbors.append(_event)
        return neighbors


    def expand_cluster(self, cluster: EventCluster, point: SDTEvent, neighbors: List[SDTEvent], points: Collection[SDTEvent]) -> EventCluster:
        """
        Expands an existing cluster from a seed point by recursively 
        adding density-reachable neighbors.

        Args:
            cluster (EventCluster): The cluster being populated.
            point (SDTEvent): The current seed point.
            neighbors (List[SDTEvent]): Initial list of neighbors for the seed.
            points (Collection[SDTEvent]): The global list of events.

        Returns:
            EventCluster: The fully expanded cluster.
        """
        point['event_status'] = CLUSTER
        cluster.add_point(point)

        index = 0
        while index < len(neighbors):
            current = neighbors[index]
            p_status = current['event_status']
            
            if p_status == UN_DETERMINED:
                current_neighbors = self.get_neighbors(current, points)
                if len(current_neighbors) >= self.min_pts:
                    # Point is a core point; merge its neighbors into the seed's list
                    merge(neighbors, current_neighbors)
                
                current['event_status'] = CLUSTER
                cluster.add_point(current)
            elif p_status == NOISE:
                 # Change noise point status to part of a cluster (border point)
                 current['event_status'] = CLUSTER
                 cluster.add_point(current)
            index += 1
            
        return cluster

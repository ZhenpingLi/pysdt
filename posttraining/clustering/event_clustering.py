import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import plugin_manager
from algorithm.subsystem_output import SubsystemOutput
from config import sdt_config
from posttraining.clustering.event_cluster import EventCluster
from posttraining.clustering.h_cluster import HCluster
from posttraining.clustering.sdt_event import SDTEvent
from posttraining.clustering.sdt_event_data import SDTEventData
from sdt_exception import SDTException
from util import time_util

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.sdt_constants import NORMAL, WARNING, ERROR, NOISE
from posttraining.clustering import event_config, sdt_event_tree
from training import data_buffer

# Constants
CONTEXT = "EventClustering"
DAY_IN_SECONDS = 86400


def build_event_data_list(node_list: List[SubsystemOutput], cutoff_time: float) -> List[SDTEventData]:
    """
    Aggregates all mnemonic-level events from a list of subsystem outputs.
    
    Filters events that occurred after the specified cutoff time and have 
    a positive intensity value.

    Args:
        node_list (List[SubsystemOutput]): The results from a training session.
        cutoff_time (float): The start timestamp for the current analysis window.

    Returns:
        List[SDTEventData]: A flat list of event data objects.
    """
    event_list: List[SDTEventData] = []
    for subsystem_output in node_list:
        for mn_output in subsystem_output.mnemonic_output_list:
            if mn_output.mnemonic_event_list:
                for sdt_event_data in mn_output.mnemonic_event_list:
                    if sdt_event_data.event_value > 0 and sdt_event_data.event_start > cutoff_time:
                        event_list.append(sdt_event_data)
    return event_list


class EventClustering:
    """
    Orchestrator for the density-based clustering of satellite anomalies.
    
    This class manages the high-level workflow of:
    1. Retrieving historical events from the archive.
    2. Building hierarchical event trees (Satellite -> Subsystem -> Mnemonic).
    3. Performing H-Clustering to identify related anomalies.
    4. Categorizing clusters (e.g., identifying recurring issues vs. noise).
    5. Exporting human-readable cluster reports.
    """
    
    COORDNAMES = ["\u03C3", "\u03C3 Diff", "Outliters"]

    def __init__(self):
        """Initializes the EventClustering orchestrator and its configuration."""
        event_config.init_event_config()
        self.long_period_event_list: Optional[List[SDTEvent]] = None
        self.ex_zone_list: Optional[List[SDTEvent]] = None
        self.sdt_data_io = None

    def analyze_sat_data(self, node_list: List[SubsystemOutput]) -> Optional[List[EventCluster]]:
        """
        Executes the full clustering analysis pipeline for the current session.
        
        It retrieves history, combines it with current session events, runs 
        the clustering algorithm, and evaluates the resulting clusters to 
        identify significant anomalies.

        Args:
            node_list (List[SubsystemOutput]): The training results for the satellite.

        Returns:
            Optional[List[EventCluster]]: A list of event clusters representing 
                identified anomalies and noise groups.
        """
        _list: List[EventCluster] = []
        if node_list:
            self.sdt_data_io = plugin_manager.get_sdt_data_training_io("default")
            cut_off_time = data_buffer.session_time
            sdt_event_list: List[SDTEvent] = []
            
            # 1. Gather current events
            event_list: List[SDTEventData] = build_event_data_list(node_list, cut_off_time)
            error_cluster = EventCluster(NOISE)
            history_start = data_buffer.session_end - event_config.event_history_limit
            
            filtered_event_list = None
            if event_list:
                event_list.sort(key=lambda c: c.event_start)
                # Separate regular events from exclusion zone events (maneuvers, etc.)
                regular_event_list, self.ex_zone_list = sdt_event_tree.split_event_list(event_list, cut_off_time)
                # Build initial hierarchical trees
                sdt_event_list, long_period_list = sdt_event_tree.build_sdt_event_list(regular_event_list)
                
                if sdt_event_list:
                    filtered_event_list = [_event for _event in sdt_event_list if _event.event_value > 0.5]
                if long_period_list:
                    self.long_period_event_list = long_period_list

            if filtered_event_list:
                # 2. Retrieve and integrate historical events
                history_event_data_list: List[SDTEventData] = []
                try:
                    history_event_data_list = self.sdt_data_io.get_event_history(history_start, cut_off_time)
                except SDTException as ex:
                    logging.info(f"{CONTEXT}: Error retrieving event history: {ex}")
                
                history_event_data_list.sort(key=lambda c: c.event_start)
                
                # Write current events to archive for future history
                if event_list:
                    try:
                        self.sdt_data_io.write_events(event_list, cut_off_time)
                    except SDTException:
                        logging.error(f"{CONTEXT}: Error writing events to data archive.")
                
                if history_event_data_list:
                    h_event_list, h_long_list = sdt_event_tree.build_sdt_event_list(history_event_data_list)
                    if h_long_list:
                        if self.long_period_event_list:
                            self.long_period_event_list.extend(h_long_list)
                        else:
                            self.long_period_event_list = h_long_list
                    if h_event_list:
                        sdt_event_list.extend(h_event_list)
                    
                # 3. Perform Clustering
                if len(sdt_event_list) > 40:
                    h_cluster = HCluster(cut_off_time)
                    event_cluster_list = h_cluster.clustering(sdt_event_list)
                    
                    if len(event_cluster_list) > 0:
                        self.output_event_cluster(event_cluster_list)
                    
                    # Refine clusters (e.g., move sparse clusters to NOISE)
                    reduced_list = self.reclustering(event_cluster_list)
                        
                    for cluster in reduced_list:
                        if cluster.get_cluster_type() == NOISE:
                            points = cluster.get_points()
                            for point in points:
                                if point['event_start'] > cut_off_time:
                                    ab_distance = point['event_value']
                                    status = NORMAL
                                    if ab_distance > event_config.event_error_threshold:
                                        status = ERROR
                                    elif ab_distance > event_config.event_warning_threshold:
                                        status = WARNING
                                            
                                    if status in (WARNING, ERROR):
                                        logging.info(f"{CONTEXT}: Significant Event Found: {point}")
                                        error_cluster.add_point(point)

            if self.sdt_data_io is not None:
                self.sdt_data_io.close()
                
            _list.append(error_cluster)
            if self.long_period_event_list:
                _list.extend(self.long_period_event_list)
        else:
            logging.warning(f"{CONTEXT}: No training output provided; analysis skipped.")
            return None
            
        return _list

    def output_event_cluster(self, event_cluster_list: List[EventCluster]):
        """
        Exports the results of the clustering analysis to a text file.
        
        The file is stored in the satellite's 'events' directory with a 
        timestamped filename.

        Args:
            event_cluster_list (List[EventCluster]): The clusters to export.
        """
        sat_id = sdt_config.sat_id
        parent_dir = Path(__file__).resolve().parent.parent.parent
        session_tag = time_util.get_simple_time_tag_from_seconds(data_buffer.session_time)
        
        event_file_name = os.path.join(parent_dir, "db", sat_id.lower(), "events", 
                                       f"{sat_id}-event-cluster-{session_tag}.txt")
        try:
            os.makedirs(os.path.dirname(event_file_name), exist_ok=True)

            with open(event_file_name, 'w') as writer:
                for event_cluster in event_cluster_list:
                    writer.write(f"Cluster: size={len(event_cluster.sdt_event_list)}, type={event_cluster.cluster_type}\n")
                    for sdt_event in event_cluster.sdt_event_list:
                        start_tag = time_util.get_time_tag_from_seconds(sdt_event.event_start)
                        end_tag = time_util.get_simple_time_tag_from_seconds(sdt_event.event_end)
                        writer.write(f"  Event: {start_tag} to {end_tag}\n")
                        for subsystem_event in sdt_event.get_children():
                            writer.write(f"    Subsystem: {subsystem_event.name}\n")
                            for mn_event in subsystem_event.get_children():
                                writer.write(f"      {mn_event.name}: val={mn_event.event_value:.2f}, state={mn_event.event_state}\n")
                    writer.write("\n")
            logging.info(f"{CONTEXT}: Event cluster report written to {event_file_name}")
        except (OSError, IOError) as e:
            logging.error(f"{CONTEXT}: Failed to write event cluster file: {e}")

    def reclustering(self, event_cluster_list: List[EventCluster]) -> List[EventCluster]:
        """
        Refines the initial clustering results by identifying sparse or 
        non-repeating clusters and re-categorizing them as NOISE.
        
        A cluster is kept only if it spans more than 'min_pts' distinct days.

        Args:
            event_cluster_list (List[EventCluster]): The initial set of clusters.

        Returns:
            List[EventCluster]: The refined list of clusters.
        """
        noise_cluster = next((c for c in event_cluster_list if c.get_cluster_type() == NOISE), None)
        reduced_list = []
        
        for cluster in event_cluster_list:
            if cluster.get_cluster_type() != NOISE:
                # Requirement: event must recur across multiple days to be a valid cluster
                if cluster.get_day_event_number() < event_config.min_pts:
                    if noise_cluster is None:
                        noise_cluster = EventCluster(NOISE)
                    for event in cluster.get_points():
                        noise_cluster.add_point(event)
                else:
                    reduced_list.append(cluster)
                    
        if noise_cluster:
            reduced_list.append(noise_cluster)
            
        return reduced_list


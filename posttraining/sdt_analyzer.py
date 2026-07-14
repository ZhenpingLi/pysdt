import logging
import os
import sys
from typing import List, Optional, Set

from algorithm.subsystem_output import SubsystemOutput
from posttraining.clustering.event_cluster import EventCluster
from posttraining.clustering.event_clustering import EventClustering
from posttraining.data_quality_metrics import evaluate_quality_metrics
from posttraining.mnemonic_status import MnemonicStatus

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from posttraining.analyzer import Analyzer
from algorithm.trend_node import TrendNode
from config.sdt_constants import NOISE
from sdtdb import sdt_db

# Constants
CONTEXT = "SDTAnalyzer"
NORMALSTATUS = 1
WARNINGSTATUS = 2
ERRORSTATUS = 3
TRAININGERROR = 4


class SDTAnalyzer(Analyzer):
    """
    Main analysis engine for post-training health and safety evaluation.
    
    This class implements the Analyzer interface. It orchestrates the entire 
    post-training workflow:
    1. Performs event clustering to identify related anomalies.
    2. Identifies 'noisy' mnemonics based on cluster patterns.
    3. Evaluates health metrics (TPC, Outliers) for every mnemonic.
    4. Generates a final status list for all satellite subsystems.
    """

    def __init__(self):
        """Initializes the SDTAnalyzer instance."""
        self.sat_id: Optional[str] = None
        self.status_string_list: List[str] = []
        self.noise_mnemonics: Set[str] = set()

    def analyze(self, data_list: List[SubsystemOutput]) -> List[MnemonicStatus]:
        """
        Executes a full analysis cycle on a set of subsystem training results.
        
        It clusters detected outliers into events, filters out noise, and 
        calculates operational status metrics for each telemetry point.

        Args:
            data_list (List[SubsystemOutput]): Aggregated outputs from a 
                training session.

        Returns:
            List[MnemonicStatus]: A list containing the health status and metrics 
                for every analyzed mnemonic.
        """
        self.status_string_list = []
        
        # 1. Event Clustering
        e_clustering = EventClustering()
        event_clusters = e_clustering.analyze_sat_data(data_list)
        self.create_noise_mnemonics(event_clusters)

        # 2. Status Evaluation
        clusters: List[EventCluster] = []
        clusters.extend(event_clusters)

        tpc_cluster = EventCluster(NOISE)
        status_list : List[MnemonicStatus] = []
        
        for subsystem_output in data_list:
            subsystem_name = subsystem_output.subsystem_name
            if not self.sat_id:
                self.sat_id = sdt_db.get_sat_id(subsystem_name)
            
            if subsystem_name != "events":
                if subsystem_output.mnemonic_output_list:
                    for mn_output in subsystem_output.mnemonic_output_list:
                        # Check if this mnemonic is part of a noise cluster
                        is_in_noise = self.is_mn_exist_in_noise_events(mn_output.mnemonic_id)
                        
                        # Evaluate health metrics
                        mn_status_list, tpc_event_list = evaluate_quality_metrics(mn_output, is_in_noise)
                        
                        if tpc_event_list:
                            tpc_cluster.sdt_event_list.extend(tpc_event_list)
                        if mn_status_list:
                            status_list.extend(mn_status_list)
                            
        if tpc_cluster.sdt_event_list:
            clusters.append(tpc_cluster)

        # Log significant error clusters
        if len(clusters) > 0:
            logging.info(f"{CONTEXT}: Significant event clusters identified:")
            for cluster in clusters:
                logging.info(f"{cluster}")
        
        return status_list


    def get_status(self, data_list: List[TrendNode], start: float, end: float) -> List[EventCluster]:
        """
        Determines current operational status for monitoring mode.
        (Placeholder implementation).

        Args:
            data_list (List[TrendNode]): The input training nodes.
            start (float): Start timestamp.
            end (float): End timestamp.

        Returns:
            List[EventCluster]: An empty list in this placeholder implementation.
        """
        return []

    def is_mn_exist_in_noise_events(self, mnemonic_id: str) -> bool:
        """
        Checks if a mnemonic was previously identified as contributing to a noise cluster.

        Args:
            mnemonic_id (str): The mnemonic to check.

        Returns:
            bool: True if identified as noise, False otherwise.
        """
        if not self.noise_mnemonics:
            return False
        return mnemonic_id in self.noise_mnemonics

    def create_noise_mnemonics(self, noise_clusters: List[EventCluster]):
        """
        Extracts mnemonic IDs from event clusters identified as NOISE and 
        populates the internal noise_mnemonics set.

        Args:
            noise_clusters (List[EventCluster]): The output from the clustering analysis.
        """
        if len(noise_clusters) > 0:
            self.noise_mnemonics = set()
            for event_cluster in noise_clusters:
                if event_cluster.get_cluster_type() == NOISE and len(event_cluster.get_points()) > 0:
                    for sdt_event in event_cluster.get_points():
                        # Navigate the hierarchy to extract mnemonic names
                        for sub_system in sdt_event.children:
                            for mn_node in sub_system.children:
                                self.noise_mnemonics.add(mn_node['name'])

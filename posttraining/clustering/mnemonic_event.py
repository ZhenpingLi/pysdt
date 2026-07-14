from __future__ import annotations

import os
import sys
from typing import List

from posttraining.clustering import event_config
from posttraining.clustering.sdt_event_data import SDTEventData

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdtdb import sdt_db
from algorithm.outlier import Outlier


def create_mnemonic_event_data_list_from_outlier_list(outlier_list: list[Outlier]) -> List[SDTEventData]:
    """
    Processes a raw list of outliers for a mnemonic and groups them into initial event data segments.
    
    It iterates through sorted outliers, calculates a normalized value using an 'ocscale' 
    attribute, and aggregates contiguous outliers into SDTEventData objects.

    Args:
        outlier_list (list[Outlier]): The list of raw outliers.

    Returns:
        List[SDTEventData]: A list of merged event data segments.
    """
    if not outlier_list:
        return []
        
    event_data_list: List[SDTEventData] = []
    outlier_list.sort(key=lambda r: r['time'])
    event_scale = event_config.SCALE
    mnemonic_id = outlier_list[0]['mnemonic_id']
    scale_str = sdt_db.get_algorithm_attribute(mnemonic_id, "ocscale")
    sample_period = sdt_db.get_mnemonic_type(mnemonic_id).frequency
    event_scale = float(scale_str)* event_scale if scale_str else event_scale
    event_start_value=0
    event_end_value=0
    
    for outlier in outlier_list:
        n_value = abs(outlier['diff'])/event_scale
        if len(event_data_list)>0:
            current_event_data = event_data_list[-1]
            if _is_joint(current_event_data, outlier['time'], sample_period):
                if outlier['time'] < current_event_data.event_start:
                    current_event_data.event_value += (event_start_value + n_value) * 0.5 * (current_event_data.event_start - outlier['time'])
                    event_start_value = n_value
                    current_event_data.event_start = outlier['time']
                elif outlier['time'] > current_event_data.event_end:
                    current_event_data.event_value += (event_end_value + n_value) * 0.5 * (outlier['time'] - current_event_data.event_end)
                    event_end_value = n_value
                    current_event_data.event_end = outlier['time']
            else:
                current_event_data = create_mnemonic_event_data_from_outlier(outlier)
                event_start_value=n_value
                event_end_value=n_value
                event_data_list.append(current_event_data)
        else:
            current_event_data = create_mnemonic_event_data_from_outlier(outlier)
            event_start_value=n_value
            event_end_value=n_value
            event_data_list.append(current_event_data)
            
    return create_merged_sdt_event_data_list(event_data_list)

def create_merged_sdt_event_data_list(event_data_list: List[SDTEventData]) -> List[SDTEventData]:
    """
    Consolidates a list of event segments by merging segments that are temporally 
    contiguous based on the global join threshold.

    Args:
        event_data_list (List[SDTEventData]): Initial list of event segments.

    Returns:
        List[SDTEventData]: The final list of merged events.
    """
    if not event_data_list:
        return []
        
    merged_event_data_list: List[SDTEventData] = []
    event_data_list.sort(key=lambda r: r.event_start)
    current_event_data = event_data_list[0]
    merged_event_data_list.append(current_event_data)
    
    for index in range(1, len(event_data_list)):
        if _is_joint(current_event_data, event_data_list[index].event_start, event_config.oc_join_time):
            if event_data_list[index].event_value > event_config.event_threshold:
                merge_sdt_event_data(current_event_data, event_data_list[index])
        elif event_data_list[index].event_value > event_config.event_threshold:
            current_event_data = event_data_list[index]
            merged_event_data_list.append(current_event_data)
            
    return merged_event_data_list

def merge_sdt_event_data(event1: SDTEventData, event2: SDTEventData):
    """
    In-place merges the second event segment into the first.

    Args:
        event1 (SDTEventData): The target event to grow.
        event2 (SDTEventData): The event to be merged.
    """
    if event1.event_value > event_config.event_threshold and event2.event_value > event_config.event_threshold:
        event1.event_start = min(event1.event_start, event2.event_start)
        event1.event_end = max(event1.event_end, event2.event_end)
        event1.event_value += event2.event_value
        return

def create_mnemonic_event_data_from_outlier(outlier: Outlier) -> SDTEventData:
    """
    Factory function to create a new SDTEventData instance from a single outlier.

    Args:
        outlier (Outlier): The source outlier.

    Returns:
        SDTEventData: The new event data object.
    """
    return SDTEventData(name=outlier['mnemonic_id'],
                        event_value=0.0,
                        event_start=outlier['time'],
                        event_end=outlier['time'],
                        event_type='outlier',
                        event_state=outlier['state'])

def _is_joint(event_data: SDTEventData, time: float, threshold: float) -> bool:
    """
    Determines if a given time is contiguous with the specified event segment.

    Args:
        event_data (SDTEventData): The existing event segment.
        time (float): The timestamp to check.
        threshold (float): The maximum allowed gap in seconds.

    Returns:
        bool: True if the time is joint with the event, otherwise False.
    """
    return (event_data.event_start <= time <= event_data.event_end) or \
        abs(event_data.event_start - time) < threshold or \
        abs(time - event_data.event_end) < threshold

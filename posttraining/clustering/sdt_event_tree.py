from math import sqrt
from typing import Optional, List, Tuple

from config import sdt_config
from posttraining.clustering import event_config
from posttraining.clustering.sdt_event import SDTEvent
from posttraining.clustering.sdt_event_data import SDTEventData
from sdtdb import sdt_db


def build_sdt_event_list(event_data_list: List[SDTEventData]) -> Tuple[Optional[List[SDTEvent]], Optional[List[SDTEvent]]]:
    """
    Constructs a list of hierarchical satellite-level events from flat event segments.
    
    It groups temporally contiguous events into trees and separates out long-duration 
    events for specific analysis.

    Args:
        event_data_list (List[SDTEventData]): The input list of flat event segments.

    Returns:
        Tuple[Optional[List[SDTEvent]], Optional[List[SDTEvent]]]: A tuple containing 
            (regular_hierarchical_events, long_period_events).
    """
    sdt_event_list = []
    long_period_event_list = []
    if event_data_list:
        for event_data in event_data_list:
            if (event_data.event_end - event_data.event_start) > event_config.event_period_limit:
                long_period_event_list.append(SDTEvent(event_data, SDTEvent.MNEMONIC, None))
            else:
                if event_data.event_value > event_config.cluster_threshold:
                    if sdt_event_list:
                        current_event :SDTEvent = sdt_event_list[-1]
                        if is_event_joint(current_event, event_data.event_start, event_data.event_end):
                            add_child_node(current_event, event_data)
                            continue
                        else:
                            current_event = create_sdt_event_tree(event_data)
                            sdt_event_list.append(current_event)
                    else:
                        current_event = create_sdt_event_tree(event_data)
                        sdt_event_list.append(current_event)
        
        return sdt_event_list if sdt_event_list else None, \
               long_period_event_list if long_period_event_list else None
    else:
        return None, None

def split_event_list(sdt_event_list: List[SDTEventData], cutoff_time: float) -> Tuple[List[SDTEventData], List[SDTEventData]]:
    """
    Splits an event list into regular events and exclusion zone events.
    
    Exclusion zone events are those belonging to the special 'events' subsystem 
    (e.g., maneuvers, dumps) that should be excluded from standard health evaluation.

    Args:
        sdt_event_list (List[SDTEventData]): The source event list.
        cutoff_time (float): Threshold to ignore old events.

    Returns:
        Tuple[List[SDTEventData], List[SDTEventData]]: (regular_events, ex_zone_events).
    """
    event_list = []
    ex_zone_list = []
    for event in sdt_event_list:
        if event.event_end > cutoff_time:
            subsystem_name = sdt_db.get_subsystem_for_mnemonic(event.name)
            if subsystem_name == 'events':
                ex_zone_list.append(event)
            else:
                event_list.append(event)
    return event_list, ex_zone_list

def create_sdt_event_tree(sdt_event_data: SDTEventData) -> SDTEvent:
    """
    Factory function to create a new hierarchical event tree for a satellite.
    
    Constructs a SATELLITE root, a SUBSYSTEM child, and a MNEMONIC leaf from 
    a single flat event data object.

    Args:
        sdt_event_data (SDTEventData): The source data.

    Returns:
        SDTEvent: The root of the newly created hierarchical event tree.
    """
    root : SDTEvent = SDTEvent(sdt_event_data, SDTEvent.SATELLITE, None)
    root['name'] = sdt_config.sat_id
    subsystem_event: SDTEvent = SDTEvent(sdt_event_data, SDTEvent.SUBSYSTEM, root)
    subsystem_event['name']= sdt_db.get_subsystem_for_mnemonic(sdt_event_data.name)
    SDTEvent(sdt_event_data, SDTEvent.MNEMONIC, subsystem_event)
    return root

def merge_sdt_event(event1: SDTEvent, event2: SDTEvent):
    """
    Recursively merges the contents of one hierarchical event tree into another.

    Args:
        event1 (SDTEvent): The target tree to be modified.
        event2 (SDTEvent): The source tree to merge from.
    """
    if event1['event_level'] == SDTEvent.MNEMONIC:
        event1['event_start'] = min(event1['event_start'], event2['event_start'])
        event1['event_end'] = max(event1['event_end'], event2['event_end'])
        event1['event_value'] += event2['event_value']
        return
        
    for sub_event2 in event2.get_children():
        sub_event1 = event1.get_child(sub_event2['name'])
        if sub_event1 is not None:
            merge_sdt_event(sub_event1, sub_event2)
        else:
            # Attach new subtree
            event1.get_children().append(sub_event2)
            sub_event2.parent = event1


def add_child_node(node: SDTEvent, event_data: SDTEventData):
    """
    Recursively adds a flat mnemonic event data object to a hierarchical tree.
    
    It updates timestamps and aggregates event values (using root-sum-square) 
    as it traverses down the levels from SATELLITE to MNEMONIC.

    Args:
        node (SDTEvent): The current hierarchical node being processed.
        event_data (SDTEventData): The mnemonic-level data to add.
    """
    node['event_start'] = min(node['event_start'], event_data.event_start)
    node['event_end'] = max(node['event_end'], event_data.event_end)
    
    if node['event_level'] == SDTEvent.MNEMONIC:
        node['event_value'] += event_data.event_value
        return
    else:
        # Aggregate intensity using RSS (Root Sum Square)
        _event_value = node['event_value']**2 + event_data.event_value**2
        node['event_value'] = sqrt(_event_value)
        
        child_name = event_data.name
        if node['event_level'] == SDTEvent.SATELLITE:
            child_name = sdt_db.get_subsystem_for_mnemonic(event_data.name)
            
        subsystem_child = node.get_child(child_name)
        if subsystem_child is not None:
            add_child_node(subsystem_child, event_data)
        else:
            # Create a new branch if the subsystem or mnemonic doesn't exist yet
            subsystem_node = SDTEvent(event_data, node['event_level'] + 1, node)
            subsystem_node['name'] = child_name
            if subsystem_node['event_level'] == SDTEvent.SUBSYSTEM:
                SDTEvent(event_data, subsystem_node['event_level'] + 1, subsystem_node)


def is_event_joint(sdt_event: SDTEvent, oc_start: float, oc_end: float) -> bool:
    """
    Checks if a hierarchical event and a new time segment are temporally contiguous.

    Args:
        sdt_event (SDTEvent): The existing hierarchical event.
        oc_start (float): The start timestamp of the candidate segment.
        oc_end (float): The end timestamp of the candidate segment.

    Returns:
        bool: True if they overlap or are within the join threshold, otherwise False.
    """
    return is_overlap(sdt_event, oc_start, oc_end)  or \
        abs(oc_start - sdt_event['event_end']) < event_config.event_join_time or \
        abs(sdt_event['event_start'] - oc_end) < event_config.event_join_time

def is_overlap(sdt_event: SDTEvent, start_time: float, end_time: float) -> bool:
    """Checks for temporal overlap between a hierarchical event and a time range."""
    return sdt_event['event_start'] <= end_time and start_time <= sdt_event['event_end']

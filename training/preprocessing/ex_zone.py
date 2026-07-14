import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from training.training_set import TrainingSet

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sdtdb import sdt_db
from algorithm.algorithm_def import AlgorithmDef
from util.time_util import get_simple_time_tag_from_seconds

# --- Constants ---
EXZONE = "exzone"
LATENCY = "latency"
ZONEPAD = "zonepad"
MINDIFF = "mindiff"
CONTEXT = "EXZONE"


def is_overlap(start: float, end: float, zone: List[float]) -> bool:
    """
    Checks if a time range overlaps with a specific exclusion zone.

    Args:
        start (float): Start of the time range.
        end (float): End of the time range.
        zone (List[float]): A list containing [zone_start, zone_end].

    Returns:
        bool: True if there is any overlap, otherwise False.
    """
    return zone[0] <= start < zone[1] or zone[0] < end <= zone[1] or start < zone[0] < zone[1] <= end


class ExZone:
    """
    Represents a collection of time intervals (zones) to be excluded from processing.
    
    This class provides robust utilities for managing exclusion regions, such as 
    maneuver periods or data dropouts. It supports set-like operations (merge, 
    intersect), temporal padding, and high-performance querying using binary search.
    """

    def __init__(self, zones: np.ndarray, mnemonic_id: str):
        """
        Initializes an ExZone object.

        Args:
            zones (np.ndarray): A 2D NumPy array where each row is [start, end].
            mnemonic_id (str): The ID of the mnemonic associated with these zones.
        """
        self.mnemonic_id = mnemonic_id
        # Ensure zones are sorted by start time for efficient binary search processing
        if zones.shape[0] > 0:
            self.ex_zones = zones[zones[:, 0].argsort()]
            self.max_dur = float(np.max(self.ex_zones[:, 1] - self.ex_zones[:, 0]))
        else:
            self.ex_zones = zones
            self.max_dur = 0.0

    def is_in_zone(self, time: float, pad_scale: float = 0.0) -> bool:
        """
        Determines if a given timestamp falls within any of the exclusion zones.
        
        Uses binary search (np.searchsorted) to ensure O(log N) lookup time, 
        making it suitable for large-scale telemetry datasets.

        Args:
            time (float): The Unix timestamp to check.
            pad_scale (float): A factor to temporarily expand the zone duration 
                for this check. Defaults to 0.0.

        Returns:
            bool: True if the time is within an excluded region, otherwise False.
        """
        if self.ex_zones.shape[0] == 0:
            return False
            
        # Find the insertion point to identify the potential containing zone
        idx = np.searchsorted(self.ex_zones[:, 0], time, side='right')
        if idx == 0:
            return False
            
        zone_start, zone_end = self.ex_zones[idx - 1]
        
        if pad_scale > 0:
            zone_end += pad_scale * (zone_end - zone_start)
            
        return zone_start <= time < zone_end

    def get_mnemonic_id(self) -> str:
        """Returns the mnemonic ID associated with these zones."""
        return self.mnemonic_id

    def intersect(self, other_zone: 'ExZone') -> Optional['ExZone']:
        """
        Calculates the temporal intersection of two ExZone sets.

        Args:
            other_zone (ExZone): The other set of zones to intersect with.

        Returns:
            Optional[ExZone]: A new ExZone containing only the overlapping 
                regions, or None if no overlap exists.
        """
        if not other_zone or self.ex_zones.shape[0] == 0 or other_zone.ex_zones.shape[0] == 0:
            return None

        intersections = []
        for z1_start, z1_end in self.ex_zones:
            for z2_start, z2_end in other_zone.ex_zones:
                if z1_end < z2_start or z2_end < z1_start:
                    continue
                overlap_start = max(z1_start, z2_start)
                overlap_end = min(z1_end, z2_end)
                if overlap_start < overlap_end:
                    intersections.append([overlap_start, overlap_end])
        
        if not intersections:
            return None
            
        return ExZone(np.array(intersections), self.mnemonic_id)

    def merge(self, other_zone: 'ExZone', is_append: bool = True):
        """
        Combines the zones from another ExZone into this instance.
        Automatically merges overlapping or adjacent results.

        Args:
            other_zone (ExZone): The source zones to merge.
            is_append (bool): Unused, maintained for signature compatibility.
        """
        if not other_zone or other_zone.ex_zones.size == 0:
            return
            
        all_zones = np.vstack([self.ex_zones, other_zone.ex_zones])
        self.ex_zones = all_zones[all_zones[:, 0].argsort()]
        self.merge_zones()

    def get_zones(self) -> np.ndarray:
        """Returns the internal 2D array of [start, end] intervals."""
        return self.ex_zones

    def get_max_dur(self) -> float:
        """Returns the duration of the longest zone in the set."""
        return self.max_dur

    def add_pad(self, pre: float, post: float):
        """
        Expands every zone in the set by the specified durations.

        Args:
            pre (float): Seconds to subtract from the start of each zone.
            post (float): Seconds to add to the end of each zone.
        """
        if self.ex_zones.shape[0] > 0:
            self.ex_zones[:, 0] -= pre
            self.ex_zones[:, 1] += post
            self.merge_zones()

    def merge_zones(self):
        """
        Internal helper to consolidate overlapping or adjacent time intervals.
        Ensures the 'ex_zones' array contains a minimal set of disjoint intervals.
        """
        if self.ex_zones.shape[0] < 2:
            return
        
        merged = []
        current_start, current_end = self.ex_zones[0]
        
        for next_start, next_end in self.ex_zones[1:]:
            if next_start <= current_end: # Overlap or touch
                current_end = max(current_end, next_end)
            else:
                merged.append([current_start, current_end])
                current_start, current_end = next_start, next_end
        
        merged.append([current_start, current_end])
        self.ex_zones = np.array(merged)

    def copy(self) -> 'ExZone':
        """Creates a deep copy of the ExZone instance."""
        return ExZone(self.ex_zones.copy(), self.mnemonic_id)

    def filter(self, min_dur: float) -> Optional['ExZone']:
        """
        Removes zones whose duration is shorter than the threshold.

        Args:
            min_dur (float): Minimum required duration in seconds.

        Returns:
            Optional[ExZone]: A new filtered ExZone, or None.
        """
        if self.ex_zones.shape[0] == 0:
            return None
            
        durations = self.ex_zones[:, 1] - self.ex_zones[:, 0]
        valid_zones = self.ex_zones[durations >= min_dur]
        
        return ExZone(valid_zones, self.mnemonic_id) if valid_zones.size > 0 else None

    def __str__(self) -> str:
        """Returns a readable summary of all exclusion segments."""
        zone_str = f"{CONTEXT} for {self.mnemonic_id}:\n"
        if self.ex_zones.size > 0:
            for i, (start, end) in enumerate(self.ex_zones):
                start_tag = get_simple_time_tag_from_seconds(start)
                end_tag = get_simple_time_tag_from_seconds(end)
                zone_str += f"  zone {i + 1}: {start_tag} | {end_tag}\n"
        else:
            zone_str += "  No zones defined."
        return zone_str

    def filter_by_time(self, start: float, end: float) -> Optional['ExZone']:
        """
        Returns a subset of zones that overlap with the specified window.

        Args:
            start (float): Window start.
            end (float): Window end.

        Returns:
            Optional[ExZone]: Sub-collection of overlapping zones.
        """
        filtered_zones = [z for z in self.ex_zones if is_overlap(start, end, z.tolist())]
        return ExZone(np.array(filtered_zones), self.mnemonic_id) if filtered_zones else None


    @staticmethod
    def create_ex_zone(mnemonic_id: str, training_set: TrainingSet) -> Optional['ExZone']:
        """
        Dynamically generates exclusion zones by detecting state changes in a mnemonic.
        
        It looks for sudden jumps in value (defined by 'mindiff') and creates 
        zones between the jump start and end, applying configured latencies.

        Args:
            mnemonic_id (str): The ID of the state indicator mnemonic.
            training_set (TrainingSet): The data set used for detection.

        Returns:
            Optional[ExZone]: The generated dynamic exclusion zones.
        """
        if not training_set or training_set.raw.size < 2:
            return None

        logging.info(f"{CONTEXT}: Detecting dynamic zones for {mnemonic_id}")
        
        lat_string = sdt_db.get_mnemonic_attribute(mnemonic_id, LATENCY)
        mindiff_str = sdt_db.get_mnemonic_attribute(mnemonic_id, MINDIFF)

        pre_zone_value = 0.0
        latency_value = 0.0
        if lat_string:
            tokens = lat_string.split('|')
            pre_zone_value = float(tokens[0])
            latency_value = float(tokens[1]) if len(tokens) == 2 else 0.0
        
        min_diff = float(mindiff_str) if mindiff_str else 0.0
        
        times = training_set.inputs[:, 0]
        values = training_set.raw
        
        zones = []
        current_start = None
        for i in range(1, len(values)):
            diff = values[i] - values[i-1]
            if diff > min_diff:
                current_start = times[i]
            elif diff < -min_diff and current_start is not None:
                zones.append([current_start - pre_zone_value, times[i] + latency_value])
                current_start = None
                
        if current_start is not None:
            zones.append([current_start - pre_zone_value, times[-1] + latency_value])

        _zone = ExZone(np.array(zones), mnemonic_id)
        _zone.merge_zones()
        return _zone

    @staticmethod
    def get_ez_zone(algorithm: AlgorithmDef, start: float, end: float) -> Optional['ExZone']:
        """
        Retrieves a configured exclusion zone from the global buffer and 
        applies algorithm-specific padding.

        Args:
            algorithm (AlgorithmDef): The algorithm definition to check.
            start (float): Analysis window start.
            end (float): Analysis window end.

        Returns:
            Optional[ExZone]: The processed exclusion zone object.
        """
        from training import data_buffer
        ex_zone_id = algorithm.get_attribute(EXZONE)
        if not ex_zone_id:
            return None
        
        ex_zone = data_buffer.get_ex_zone(ex_zone_id, start, end, None)
        
        if ex_zone is not None:
            zone_pad = algorithm.get_attribute(ZONEPAD)
            if zone_pad:
                temp_z = ex_zone.copy()
                tokens = zone_pad.split('|')
                pre_pad = float(tokens[0])
                post_pad = float(tokens[1]) if len(tokens) == 2 else 0.0
                temp_z.add_pad(pre_pad, post_pad)
                ex_zone = temp_z
                
        return ex_zone

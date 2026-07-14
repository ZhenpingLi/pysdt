import logging
import os
import sys
from typing import List, Optional

import numpy as np

from sdtdb.sdt_db import StateType

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import sdt_config
from algorithm.algorithm_def import AlgorithmDef
from .ex_zone import ExZone
from orbit import orbit_model_manager
from training import data_buffer as db


# --- Constants ---
PREV = 0
CURRENT = 1
DISJOINT = "disjoint"
HOUR_IN_SECONDS = 3600
MNVR = "mnvr"
ECL = "ecl"

class StateZones:
    """
    Manager for operational state time intervals.
    
    This class is responsible for calculating and initializing the time 
    windows (zones) during which specific satellite states (e.g., maneuvers, 
    eclipses) were active. It supports complex zone logic including merging, 
    intersecting, and calculating disjoint 'default' regions.
    """
    CONTEXT = "STATEZONES"

    def __init__(self, algorithm: AlgorithmDef, mnemonic_id: str):
        """
        Initializes the StateZones manager.

        Args:
            algorithm (AlgorithmDef): The algorithm definition containing 
                state configurations.
            mnemonic_id (str): The ID of the mnemonic being processed.
        """
        self.algorithm = algorithm
        self.state_array: List[StateType] = algorithm.al_type.state or []
        self.mnemonic_id = mnemonic_id
        
        s_dur = algorithm.get_attribute("mindur")
        self.min_dur = float(s_dur) if s_dur else (sdt_config.get_float_property("MINSTATEDUR") or 900.0)

    def init_state_zones(self, session_index: int) -> List[Optional[ExZone]]:
        """
        Calculates state zones for either the current or previous session.

        Args:
            session_index (int): PREV (0) or CURRENT (1).

        Returns:
            List[Optional[ExZone]]: A list of exclusion zones, one for each 
                defined state.
        """
        if session_index == PREV:
            return self._init_zones_for_prev_session()
        else:
            return self._init_zones_for_current_session()

    def _init_zones_for_current_session(self) -> List[Optional[ExZone]]:
        """Internal helper to initialize current session zones."""
        model_time = db.get_default_model_time(self.algorithm)
        if not model_time:
             return []
        pattern_times = model_time.get_pattern_times()
        return self._calculate_state_zones(pattern_times, model_time.get_reference_time(), model_time.get_model_period())

    def _init_zones_for_prev_session(self) -> List[Optional[ExZone]]:
        """Internal helper to initialize previous session zones."""
        model_time = orbit_model_manager.create_default_model_time_for_prev_session(self.algorithm)
        if not model_time:
             return []
        pattern_times = model_time.get_pattern_times()
        return self._calculate_state_zones(pattern_times, model_time.get_reference_time(), model_time.get_model_period())

    def _calculate_state_zones(self, pattern_times: np.ndarray, reference_time: float, pattern_period: float) -> List[Optional[ExZone]]:
        """
        Core logic to identify and process time regions for every state.
        
        It parses the 'flag' and 'pad_factor' from each StateType, retrieves 
        base zones from the buffer, and applies logical merge/intersect 
        operations to define the final active regions.

        Args:
            pattern_times (np.ndarray): Nominal pattern cycles.
            reference_time (float): Reference start timestamp.
            pattern_period (float): Nominal cycle duration.

        Returns:
            List[Optional[ExZone]]: The final list of active state regions.
        """
        if not self.state_array or pattern_times is None or pattern_times.size == 0:
            return []

        state_zones: List[Optional[ExZone]] = [None] * len(self.state_array)
        end_zone = pattern_times[-1] + HOUR_IN_SECONDS / 2
        start_zone = reference_time - HOUR_IN_SECONDS / 2

        for i, state_type in enumerate(self.state_array):
            if i == 0 or state_type.flag is None:
                continue

            pad_factor = None
            if state_type.pad_factor:
                factors = state_type.pad_factor.split('|')
                if len(factors) >= 2:
                    pad_factor = (float(factors[0]), float(factors[1]))

            flags = state_type.flag
            is_intersect = "$" in flags
            zone_ids = flags.split('$') if is_intersect else flags.split('|')
            
            # Step 1: Retrieve the primary zone for this state
            zone = db.get_ex_zone(zone_ids[0], start_zone, end_zone, pad_factor)
            if zone is not None:
                # Apply minimum duration filters for maneuvers and eclipses
                if state_type.name in [MNVR, ECL]:
                    zone = zone.filter(self.min_dur)
                if not zone:
                    continue
                    
                # Step 2: Merge or Intersect additional flags
                for j in range(1, len(zone_ids)):
                    is_append = zone_ids[j] != "GNC_NAV_DV_BURN_DUR"
                    next_zone = db.get_ex_zone(zone_ids[j], reference_time, end_zone, None)
                    if next_zone:
                        if is_intersect:
                            zone = zone.intersect(next_zone)
                            if not zone: break
                        else:
                            zone.merge(next_zone, is_append=is_append)
            
                if zone is None or zone.ex_zones.size == 0:
                    continue

                if pad_factor:
                    zone.add_pad(pad_factor[0], pad_factor[1])
                    
                state_zones[i] = zone
                logging.info(f"[{self.CONTEXT}] Defined zone for state '{state_type.name}': {zone}")

        # Step 3: Handle the 'disjoint' flag (usually for the first/default state)
        if self.state_array and self.state_array[0] is not None and self.state_array[0].flag == DISJOINT:
            self._calculate_disjoint_zone(state_zones, pattern_period, pattern_times)

        return state_zones

    def _calculate_disjoint_zone(self, state_zones: List[Optional[ExZone]], pattern_period: float, pattern_times: np.ndarray):
        """
        Calculates the inverse zone for a 'default' state.
        
        The disjoint zone covers all time periods within the session that are 
        NOT covered by any other specifically defined operational states.

        Args:
            state_zones (List[Optional[ExZone]]): The list of existing zones.
            pattern_period (float): Pattern duration.
            pattern_times (np.ndarray): Pattern boundaries.
        """
        # Combine all non-default state zones
        combined_zone = None
        for i in range(1, len(state_zones)):
            if state_zones[i]:
                if combined_zone is None:
                    combined_zone = state_zones[i].copy()
                else:
                    combined_zone.merge(state_zones[i])
        
        if combined_zone is None:
            # Default state occupies the full training window
            state_zones[0] = ExZone(np.array([[pattern_times[0], pattern_times[-1]]]), self.mnemonic_id)
            return

        # Perform temporal inversion
        zones = combined_zone.get_zones()
        disjoint_list = []
        current_time = pattern_times[0]
        
        for start, end in zones:
            if current_time < start:
                disjoint_list.append([current_time, start - 0.1])
            current_time = max(current_time, end)
            
        if current_time < pattern_times[-1]:
            disjoint_list.append([current_time, pattern_times[-1]])
            
        if disjoint_list:
            state_zones[0] = ExZone(np.array(disjoint_list), self.mnemonic_id)
            logging.info(f"[{self.CONTEXT}] Calculated disjoint default zone: {state_zones[0]}")
        else:
            state_zones[0] = None

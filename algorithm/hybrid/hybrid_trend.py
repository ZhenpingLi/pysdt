import bisect
import os
import sys
from typing import List, Optional

import numpy as np

from algorithm.algorithm_data import AlgorithmData
from algorithm.hybrid_state_data import HybridStateData
from algorithm.single_state_data import SingleStateData
from config.sdt_constants import HYBRID, ECL
from sdtdb.sdt_db import StateType

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_trend import DataTrend
from training.preprocessing.state_zones import StateZones
from training.preprocessing.ex_zone import ExZone
import plugin_manager

# --- Constants ---
CONTEXT = "HybridTrend"


class HybridTrend(DataTrend):
    """
    Composite predictive model for multi-state (Hybrid) telemetry systems.
    
    A HybridTrend acts as a container for multiple DataTrend objects, each 
    optimized for a specific operational state (e.g., NORMAL, MANEUVER, 
    ECLIPSE). It dispatches prediction and evaluation requests to the 
    appropriate sub-model based on the provided timestamp.
    """

    def __init__(self, mnemonic_id: str):
        """
        Initializes a new HybridTrend model.

        Args:
            mnemonic_id (str): Unique identifier for the telemetry mnemonic.
        """
        super().__init__(mnemonic_id)
        self.trends: List[DataTrend] = []
        self.state_zones: Optional[List[ExZone]] = None
        self.model_param_dim: int = 0
        self.state_zones_obj = StateZones(self.algorithm, self.mnemonic_id)
        self.alg_name = HYBRID
        
        # Binary search optimization structures for get_state_index
        self._interval_starts: List[float] = []
        self._interval_ends: List[float] = []
        self._interval_indices: List[int] = []

    def _init_trends(self):
        """
        Initializes the sub-trend model objects based on defined states and zones.
        
        It creates a separate DataTrend instance for each active region of every 
        state, ensuring that the model can capture state-specific behaviors 
        independently.
        """
        if not self.state_zones:
            return

        state_array = self.algorithm.al_type.state
        if not state_array:
            return
            
        self.trends = []
        
        for i, zone_group in enumerate(self.state_zones):
            state_type = state_array[i]
            
            if i == 0: # Default/Baseline State
                if zone_group is None:
                    trend = self._create_trend_for_state(state_type)
                    trend.set_data_model_time(self.pattern_times, self.pattern_period)
                    self.trends.append(trend)
                else:
                    for zone in zone_group.get_zones():
                        trend = self._create_trend_for_state(state_type)
                        trend.set_pattern_period(self.pattern_period)
                        trend.set_pattern_times(zone)
                        self.trends.append(trend)
            else: # Operational/Excluded States
                if zone_group is not None:
                    dim_pointer = 0
                    if state_type.dim_pointer:
                        dim_pointer = int(state_type.dim_pointer)
                    zones = zone_group.get_zones()
                    
                    if dim_pointer is not None and dim_pointer >= 1 and state_type.name != ECL:
                        for zone in zones:
                            trend = self._create_trend_for_state(state_type)
                            trend.set_pattern_period(self.pattern_period)
                            trend.set_pattern_times(zone)
                            self.trends.append(trend)
                    else:
                        trend = self._create_trend_for_state(state_type)
                        p_times = zones.flatten()
                        trend.set_data_model_time(p_times, self.pattern_period)
                        self.trends.append(trend)

        if self.is_disjoint() and len(self.trends) > 1:
            self.trends.sort(key=lambda t: t.get_reference_time())
            
        # Re-build the index for O(log N) state lookup
        self._build_interval_index()

    def _build_interval_index(self):
        """
        Constructs sorted lookup tables for state time intervals.
        
        This enables efficient binary search identification of which operational 
        state is active at any given timestamp.
        """
        intervals = []
        
        # Collect all non-default intervals
        for i, trend in enumerate(self.trends):
            if i == 0:
                continue 
            
            p_times = trend.get_pattern_times()
            if p_times is None:
                continue
            
            num_patterns = len(p_times) // 2
            for j in range(num_patterns):
                start = p_times[2 * j]
                end = p_times[2 * j + 1]
                intervals.append((start, end, i))
        
        intervals.sort(key=lambda x: x[0])
        
        if intervals:
            self._interval_starts = [x[0] for x in intervals]
            self._interval_ends = [x[1] for x in intervals]
            self._interval_indices = [x[2] for x in intervals]
        else:
            self._interval_starts = []
            self._interval_ends = []
            self._interval_indices = []

    def _create_trend_for_state(self, state_type: StateType) -> DataTrend:
        """Helper to instantiate a state-specific sub-trend."""
        trend = plugin_manager.get_data_trend(state_type.algorithm, self.mnemonic_id)
        trend.set_state(state_type)
        return trend

    def set_state_zones(self, state_zones: List[ExZone]):
        """
        Configures the active time intervals for each state and triggers 
        sub-trend initialization.

        Args:
            state_zones (List[ExZone]): List of exclusion zones per state.
        """
        self.state_zones = state_zones
        self._init_trends()

    def get_state_trend(self, state_name: str) -> Optional[DataTrend]:
        """Retrieves the sub-model for a specifically named state."""
        for trend in self.trends:
            if trend.get_state() == state_name:
                return trend
        return None

    def get_state_name(self, time: float) -> str:
        """Identifies the operational state name active at a given time."""
        index = self.get_state_index(time)
        return self.trends[index].get_state()

    def get_state_index(self, time: float) -> int:
        """
        Uses binary search to find the index of the trend model active at 'time'.
        
        Returns 0 (the default state) if no specific operational state 
        interval is found.

        Args:
            time (float): The Unix timestamp to query.

        Returns:
            int: The index into the internal 'trends' list.
        """
        if not self._interval_starts:
            return 0
            
        idx = bisect.bisect_right(self._interval_starts, time)
        if idx == 0:
            return 0 
            
        candidate_idx = idx - 1
        if time < self._interval_ends[candidate_idx]:
            return self._interval_indices[candidate_idx]
            
        return 0

    def get_trend_value(self, time: List[float]) -> float:
        """
        Calculates the predicted trend value using the appropriate sub-model.

        Args:
            time (List[float]): Input feature vector (time, ...).

        Returns:
            float: The predicted value from the active state model.
        """
        state_index = self.get_state_index(time[0])
        return self.trends[state_index].get_trend_value(time)

    def get_data_trends(self) -> List[DataTrend]:
        """Returns the list of all sub-trend components."""
        return self.trends

    def get_trend_values_at(self, time: List[float], post_fix: str) -> List[float]:
        """Dispatches multi-value prediction request to the active sub-model."""
        state_index = self.get_state_index(time[0])
        return self.trends[state_index].get_trend_values_at(time, post_fix)

    def get_trend_value_with_postfix(self, time: List[float], post_fix: str) -> float:
        """Dispatches specific boundary prediction request to the active sub-model."""
        state_index = self.get_state_index(time[0])
        return self.trends[state_index].get_trend_value_with_postfix(time, post_fix)

    def get_model_param_dim(self) -> int:
        """Calculates the total dimension required to serialize the full hybrid model."""
        total_dim = 1 + len(self.trends)
        if self.pattern_times is not None:
            total_dim += 2 * len(self.pattern_times)
        for trend in self.trends:
            total_dim += trend.get_model_param_dim()
        return total_dim

    def is_disjoint(self) -> bool:
        """Returns True if any sub-trend is marked as disjoint."""
        return any(trend.is_disjoint for trend in self.trends)
        
    def get_pattern_times(self) -> Optional[np.ndarray]:
        """Calculates the total time range covered by all sub-trends."""
        if not self.trends:
            return None
        
        all_times = [t.get_pattern_times() for t in self.trends if t.get_pattern_times() is not None]
        if not all_times:
            return None
            
        min_time = min(t[0] for t in all_times)
        max_time = max(t[-1] for t in all_times)
        
        return np.array([min_time, max_time])

    def reset_pattern_times(self) -> None:
        """Refreshes pattern boundaries for sub-models based on updated state zones."""
        states = self.algorithm.al_type.state
        for i, (state_zone, state) in enumerate(zip(self.state_zones, states)):
            if state_zone:
                zones = state_zone.get_zones()
                _pattern_times = [zone for zone in zones]
                state_name = state.name
                for trend in self.trends:
                    if trend.get_state() == state_name and state_name == ECL:
                        trend.set_pattern_times(_pattern_times)
        
        self._build_interval_index()

    def get_algorithm_data(self) -> AlgorithmData:
        """
        Serializes the full hybrid state into a HybridStateData object for archive.

        Returns:
            HybridStateData: The serialized composite model.
        """
        single_state_list: List[SingleStateData] = [trend.get_algorithm_data() for trend in self.trends]
        _state_zone_list = []
        if self.state_zones:
            for zones in self.state_zones:
                if zones is not None:
                    _state_zone_list.append(zones.ex_zones)
        return HybridStateData(
            mnemonic_id=self.mnemonic_id,
            alg_name=self.alg_name,
            state_zones = _state_zone_list,
            data_trend_list = single_state_list
        )

    def set_algorithm_data(self, algorithm_data: AlgorithmData):
        """
        Reconstructs the composite hybrid model from serialized archive data.

        Args:
            algorithm_data (AlgorithmData): The source HybridStateData.
        """
        hybrid_state_data : HybridStateData = algorithm_data
        self.alg_name = hybrid_state_data.alg_name
        self.trends : List[DataTrend] = [None] * len(hybrid_state_data.data_trend_list)
        for index, single_state_data in enumerate(hybrid_state_data.data_trend_list):
            if single_state_data is not None:
                state_type = self.algorithm.get_state(single_state_data.state)
                self.trends[index] = plugin_manager.get_data_trend_from_output(single_state_data)
                self.trends[index].set_state(state_type)
        self._build_interval_index()

import os
import sys
from abc import ABC, abstractmethod
from typing import List, Optional

import math
import numpy as np

from algorithm.algorithm_data import AlgorithmData
from algorithm.single_state_data import SingleStateData
from config.sdt_constants import HOUR_IN_SECONDS, EARTHYEAR_IN_SECONDS, SIGMA_INDEX, MAX, MAX_INDEX, MIN, MIN_INDEX, \
    MEAN, MEAN_INDEX, SIGMA, DEFAULT, DISJOINT, ECL
from sdtdb.sdt_db import StateType

# Add parent directory to path to find sdt_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.sdt_config as sdt_config
from sdtdb import sdt_db

from algorithm.data_point import DataPoint
from .algorithm_def import AlgorithmDef

LONGTERM = 1
TREND = "trend"
ELIMIT = "elimit"
WLIMIT = "wlimit"
WELIMIT = "welimit"
STDDEV = "stddev"

INFO = "INFO"
WARNING = "WARNING"
SOCOEF = "socoef"
STAT = "stat"

class DataTrend(ABC):
    """
    Abstract base class defining the data structure for a telemetry data trend.
    
    Contains parameters for predicting dataset values, standard deviations, 
    and handles periodic time normalization and statistical evaluation.
    """
    TREND_KEYS : List[str] = ['mnemonic_id', 'alg_name', 'pattern_period', 'ref_time', 'sigma', 'tpc', 'params_dim', 'frequency', 'time_scale', 'data_gap']
    
    def __init__(self, mnemonic_id: str):
        """
        Initializes a new instance of the DataTrend class.

        Args:
            mnemonic_id (str): The unique identifier for the telemetry mnemonic.
        """
        self.mnemonic_id :str = mnemonic_id
        self.alg_name : str = None
        from training import data_buffer as data_buffer
        self.training_type : int = data_buffer.session_type
        self.pattern_period : float = 0.0
        self.pattern_times: Optional[np.ndarray] = None
        self.ref_time : float = 0.0
        self.stat_list: Optional[List[DataPoint]] = None
        self.scale_offset_list: Optional[List[DataPoint]] = None
        self.sigma : float= 0.0
        self.tpc : float= 0.0
        self.limits : List[float] = [0.0, 0.0]
        self.algorithm: Optional[AlgorithmDef] = None
        self.params: Optional[List[float]] = None
        self.param_dim : int = 0
        self.is_monitor : bool = False
        self.is_trended : bool= False
        self.frequency : float= 0.0
        self.num_pattern_in_training : int = 0
        self.time_scale : float= HOUR_IN_SECONDS
        self.state: Optional[StateType] = None
        self.data_gap : float= 0.0
        
        if self.training_type == LONGTERM:
            self.time_scale = EARTHYEAR_IN_SECONDS
        self.init_trend()

    def init_trend(self):
        """
        Initializes the data trend by setting default values and retrieving 
        configuration from the database and system configuration.
        """
        self.spc = 0.0
        el_type = sdt_db.get_mnemonic_type(self.mnemonic_id)
        if el_type:
            self.limits[0] = el_type.warning_limit
            self.limits[1] = el_type.error_limit
            self.frequency = el_type.frequency * 1.2
        
        self.is_monitor = False
        self.is_trended = False
        algorithm_type = sdt_db.get_algorithm(self.mnemonic_id)
        if algorithm_type:
            self.algorithm = AlgorithmDef(algorithm_type)
        else:
            self.algorithm = None 
        
        num_pattern_str = sdt_config.get_config_value("NUMPATTERNINTRAINING")
        self.num_pattern_in_training = int(num_pattern_str) if num_pattern_str else 1
        
        if self.algorithm:
            _np = self.algorithm.get_np()
            if _np > 1:
                self.num_pattern_in_training = _np

    def set_data_model_time(self, pattern_times: np.ndarray, pattern_period: float):
        """
        Configures the time parameters for the data model.

        Args:
            pattern_times (np.ndarray): Array of timestamps defining the start of each pattern period.
            pattern_period (float): The duration of a single pattern cycle in seconds.
        """
        self.set_pattern_times(pattern_times)
        self.set_pattern_period(pattern_period)

    def get_alg_name(self) -> str:
        """Returns the name of the algorithm used for this trend."""
        return self.alg_name

    def get_state(self) -> str:
        """
        Returns the name of the operational state associated with this trend.
        Returns 'DEFAULT' if no state is defined.
        """
        if self.state is None:
            return DEFAULT
        else:
            return self.state.name

    def get_num_pattern_in_training(self) -> int:
        """Returns the number of pattern cycles included in the training session."""
        return self.num_pattern_in_training

    def get_reference_time(self) -> float:
        """Returns the reference timestamp for the start of the trend."""
        return self.ref_time

    def set_pattern_times(self, p_t: np.ndarray):
        """
        Sets the pattern start times.

        Args:
            p_t (np.ndarray): An array of timestamps.
        """
        if p_t is not None and p_t.size > 0:
            self.ref_time = p_t[0]
            self.pattern_times = p_t

    def get_pattern_times(self) -> Optional[np.ndarray]:
        """Returns the array of pattern start times."""
        return self.pattern_times

    def get_pattern_period(self) -> float:
        """Returns the duration of the pattern period in seconds."""
        return self.pattern_period

    def set_pattern_period(self, p_p: float):
        """
        Sets the pattern period duration.

        Args:
            p_p (float): Period in seconds.
        """
        self.pattern_period = p_p

    def get_mnemonic_id(self) -> str:
        """Returns the mnemonic identifier string."""
        return self.mnemonic_id

    def get_scale_offset_list(self) -> List[DataPoint]:
        """Returns the list of scale and offset data points used for normalization."""
        return self.scale_offset_list

    def set_scale_offset_list(self, so_list: List[DataPoint]):
        """Sets the list of scale and offset data points."""
        self.scale_offset_list = so_list

    def get_data_model_time(self, time: float) -> float:
        """
        Normalizes an absolute timestamp to a relative time within a single 
        pattern cycle, expressed in hours (or years for long-term trends).

        Args:
            time (float): The absolute timestamp in seconds.

        Returns:
            float: The normalized model time.
        """
        if self.pattern_times is None or self.pattern_times.size == 0 or self.pattern_period == 0:
            return 0.0
        
        pattern_times = self.pattern_times.astype(float)
            
        time_since = time - pattern_times[0]
        if time_since < 0:
            time_since += math.ceil(-time_since / self.pattern_period) * self.pattern_period
        elif time > pattern_times[-1]:
            time_since = time - pattern_times[-1]
            if time_since > self.pattern_period:
                num = int(time_since / self.pattern_period)
                time_since -= num * self.pattern_period
        else:
            idx = np.searchsorted(pattern_times, time, side='right') - 1
            if idx >= 0:
                time_since = time - pattern_times[idx]

        return float(time_since / self.time_scale)

    def get_all_trend_values(self, inputs: List[List[float]], post_fix: str) -> np.ndarray:
        """
        Calculates trend and limit values for a list of input time points.
        Uses vectorized operations for efficiency.

        Args:
            inputs (List[List[float]]): A list of input vectors (typically containing time).
            post_fix (str): Specifies what to calculate ('trend', 'wlimit', 'elimit', etc.).

        Returns:
            np.ndarray: A matrix containing the requested values.
        """
        trend_values = np.array([self.get_trend_value(row) for row in inputs])
        
        warning_limits = self.sigma * self.limits[0]
        error_limits = self.sigma * self.limits[1]
        
        if post_fix == WELIMIT:
            return np.column_stack([
                trend_values + error_limits,
                trend_values + warning_limits,
                trend_values,
                trend_values - warning_limits,
                trend_values - error_limits
            ])
        elif post_fix == WLIMIT:
            return np.column_stack([
                trend_values + warning_limits,
                trend_values,
                trend_values - warning_limits
            ])
        elif post_fix == ELIMIT:
            return np.column_stack([
                trend_values + error_limits,
                trend_values,
                trend_values - error_limits
            ])
        elif post_fix == SIGMA:
            return np.full((len(inputs), 1), self.sigma)
        else:
            return trend_values.reshape(-1, 1)

    def set_state(self, state: StateType):
        """Sets the operational state for this trend."""
        self.state = state

    def set_data_gap(self, gap: float):
        """Sets the percentage of missing data in the training set."""
        self.data_gap = gap

    def get_data_gap(self) -> float:
        """Returns the data gap percentage."""
        return self.data_gap

    def set_monitor(self, m: bool):
        """Enables or disables monitoring mode."""
        self.is_monitor = m

    def set_trended(self, t: bool):
        """Sets whether the model has been successfully trended (trained)."""
        self.is_trended = t

    def is_trended_check(self) -> bool:
        """Returns True if the trend model is trained and valid."""
        return self.is_trended

    def is_in_range(self, time: float) -> bool:
        """
        Checks if a given timestamp falls within the valid range of the model.

        Args:
            time (float): The timestamp to check.
        """
        if self.ref_time == 0.0: return False
        end_range = self.ref_time + 2 * self.num_pattern_in_training * self.pattern_period
        if self.is_monitor:
            end_range += self.num_pattern_in_training * self.pattern_period
        return self.ref_time <= time <= end_range

    def set_stats(self, stats: List[DataPoint]):
        """Sets the list of statistical data points (mean, max, min, sigma)."""
        self.stat_list = stats

    def get_sigma_t(self) -> float:
        """Returns the maximum historical standard deviation from the statistics list."""
        sigma_t = 0.0
        if self.stat_list:
            for d_p in self.stat_list:
                if d_p and d_p.data.size > SIGMA_INDEX:
                    sigma_t = max(sigma_t, float(d_p.data[SIGMA_INDEX]))
        return sigma_t

    def get_stat_index(self, time: float) -> int:
        """
        Finds the index in the stat_list corresponding to the given time.

        Args:
            time (float): The absolute timestamp.
        """
        if self.ref_time == 0.0 or self.pattern_period == 0.0: return 0
        diff = time - self.ref_time
        index = int(round(diff / self.pattern_period))
        
        if self.stat_list:
            if 0 <= index < len(self.stat_list):
                return index
            elif index >= len(self.stat_list):
                return len(self.stat_list) - 1
        return 0

    def get_params(self) -> List[float]:
        """Returns the trained model parameters (coefficients)."""
        if self.params is None:
            self.params = [0.0] * self.param_dim
        return self.params

    def set_params(self, p: List[float]):
        """Sets the model parameters and marks the model as trended."""
        self.params = p
        self.is_trended = True

    def set_model_params(self, p: List[float], ref_time: float):
        """
        Parses a flat parameter array and populates the trend model properties.

        Args:
            p (List[float]): Flat array containing sigma, weights, period, and pattern times.
            ref_time (float): The reference timestamp.
        """
        offset = 0
        self.sigma = p[0]
        offset += 1
        self.params = p[1:self.param_dim + 1]
        offset += self.param_dim + 1
        if len(p) > offset:
            self.pattern_period = p[self.param_dim + 1]
            self.pattern_times = np.array(p[self.param_dim +2:])
        self.ref_time = ref_time

    def get_model_params(self) -> List[float]:
        """Returns a flat list representation of all model parameters for storage."""
        s_array = [0]*self.get_model_param_dim()
        s_array[0] = self.sigma
        if self.params is not None:
            s_array[1:self.param_dim + 1] = self.params
        else:
            s_array[1:self.param_dim + 1] = [0.0] * self.param_dim
        s_array[self.param_dim + 1] = self.pattern_period
        s_array[self.param_dim + 2:] = self.pattern_times[1:]
        return s_array

    def retrieve_data_point_array(self, p: List[float], offset: int, data_type: str) -> int:
        """
        Reconstructs DataPoint lists (stats or scale/offsets) from a flat parameter array.

        Args:
            p (List[float]): The source flat array.
            offset (int): Starting index in the array.
            data_type (str): Either 'stat' or 'socoef'.

        Returns:
            int: The number of elements consumed from the array.
        """
        array_dim = 4
        if data_type == SOCOEF:
            array_dim = 2
            
        num_sets = len(self.pattern_times) - 1 if self.pattern_times is not None else 0
        if self.state and self.get_state() != DEFAULT:
            num_sets = len(self.pattern_times)//2 if self.pattern_times is not None else 0
            
        dp_list = []
        index = offset
        
        for i in range(num_sets):
            stat_list_for_dp = []
            for j in range(array_dim):
                if index < len(p):
                    stat_list_for_dp.append(float(p[index]))
                    index += 1
                else:
                    stat_list_for_dp.append(0.0)
            _ptime = self.pattern_times.astype(float)
            if self.pattern_times is not None and i < len(self.pattern_times):
                dp_list.append(DataPoint(_ptime[i], np.array(stat_list_for_dp)))

        if data_type == STAT:
            self.stat_list = dp_list
        else:
            self.scale_offset_list = dp_list
            
        return len(dp_list) * array_dim

    def get_model_param_dim(self) -> int:
        """Returns the total length of the flat parameter representation."""
        model_param_dim=self.param_dim
        if self.state is None or self.state.name==DEFAULT:
            model_param_dim += 2*self.num_pattern_in_training+1
        else:
            if self.state.name==ECL:
                model_param_dim += 4*self.num_pattern_in_training
            else:
                model_param_dim += 2
        return model_param_dim

    def get_limits(self) -> List[float]:
        """Returns the warning and error limit multipliers."""
        return self.limits

    def get_stat_at_time(self, time: float) -> Optional[DataPoint]:
        """
        Finds the statistical data point valid for a specific timestamp.

        Args:
            time (float): The absolute timestamp.
        """
        if self.stat_list and len(self.stat_list) > 0:
            times = np.array([dp.time for dp in self.stat_list if dp])
            if times.size == 0: return None
            
            idx = np.searchsorted(times, time, side='right') - 1
            idx = np.clip(idx, 0, len(self.stat_list) - 1)
            return self.stat_list[idx]
        else:
            return None

    def set_stddev(self, _sigma: float, prev_sigma: float):
        """
        Sets the standard deviation and calculates the temporal change (TPC).

        Args:
            _sigma (float): The current standard deviation.
            prev_sigma (float): The previous session's standard deviation.
        """
        self.sigma = _sigma
        if prev_sigma > 0 and _sigma != float('inf'):
            self.tpc = self.sigma / prev_sigma
        else:
            self.tpc = 0.0

    def get_stddev(self) -> float:
        """Returns the standard deviation (sigma)."""
        return self.sigma

    def get_tpc(self) -> float:
        """Returns the temporal change ratio."""
        return self.tpc

    def get_warning_limit(self, time: float) -> float:
        """Returns the absolute warning threshold value."""
        return self.sigma * self.limits[0]

    def get_error_limit(self, time: float) -> float:
        """Returns the absolute error threshold value."""
        return self.sigma * self.limits[1]

    def get_stat_list(self) -> List[DataPoint]:
        """Returns the list of statistical DataPoints."""
        return self.stat_list

    def get_stat(self, post_fix: str) -> float:
        """
        Calculates an aggregate statistic across all cycles in the session.

        Args:
            post_fix (str): 'max', 'min', 'mean', or 'sigma'.
        """
        if not self.stat_list:
            return 0.0
            
        if post_fix == MAX:
            return max(float(dp.data[MAX_INDEX]) for dp in self.stat_list if dp and dp.data.size > MAX_INDEX)
        elif post_fix == MIN:
            return min(float(dp.data[MIN_INDEX]) for dp in self.stat_list if dp and dp.data.size > MIN_INDEX)
        elif post_fix == MEAN:
            values = [dp.data[MEAN_INDEX] for dp in self.stat_list if dp and dp.data.size > MEAN_INDEX]
            return np.mean(values) if values else 0.0
        elif post_fix == SIGMA:
            values = [dp.data[SIGMA_INDEX] for dp in self.stat_list if dp and dp.data.size > SIGMA_INDEX]
            return np.mean(values) if values else 0.0
        else:
            return 0.0

    def get_trend_value_with_postfix(self, time: List[float], post_fix: str) -> float:
        """
        Returns a single scalar value (trend or limit) for a specific time and type.

        Args:
            time (List[float]): Input vector.
            post_fix (str): Value type indicator.
        """
        if post_fix == TREND:
            return self.get_trend_value(time)
        elif post_fix == ELIMIT:
            return self.get_error_limit(time[0])
        elif post_fix == WLIMIT:
            return self.get_warning_limit(time[0])
        elif post_fix == STDDEV:
            return self.sigma
        else:
            return self.get_stat(post_fix)

    def get_trend_values_at(self, time: List[float], post_fix: str) -> List[float]:
        """
        Returns a list of values (e.g., [upper, trend, lower]) for a specific time.

        Args:
            time (List[float]): Input vector.
            post_fix (str): Indicator for multiple values (e.g., 'welimit').
        """
        num = 1
        if post_fix == WELIMIT: num = 5
        elif post_fix == WLIMIT or post_fix == ELIMIT: num = 3
        
        values = [0.0] * num
        trend_value = self.get_trend_value(time)
        error_limit = self.get_error_limit(time[0])
        warning_limit = self.get_warning_limit(time[0])
        
        if post_fix == WELIMIT:
            values[0] = trend_value + error_limit
            values[1] = trend_value + warning_limit
            values[2] = trend_value
            values[3] = trend_value - warning_limit
            values[4] = trend_value - error_limit
        elif post_fix == WLIMIT:
            values[0] = trend_value + warning_limit
            values[1] = trend_value
            values[2] = trend_value - warning_limit
        elif post_fix == ELIMIT:
            values[0] = trend_value + error_limit
            values[1] = trend_value
            values[2] = trend_value - error_limit
        elif post_fix == SIGMA:
            values[0] = self.sigma
        else:
            values[0] = trend_value
            
        return values

    @abstractmethod
    def get_trend_value(self, time: List[float]) -> float:
        """
        Abstract method to calculate the predicted trend value at a given time.
        Must be implemented by subclasses.

        Args:
            time (List[float]): The input vector.
        """
        pass

    def get_param_dim(self) -> int:
        """Returns the number of trained model parameters."""
        return self.param_dim

    def get_algorithm(self) -> AlgorithmDef:
        """Returns the algorithm definition."""
        return self.algorithm

    def get_frequency(self) -> float:
        """Returns the mnemonic frequency."""
        return self.frequency

    def get_scale_offset(self, time: float) -> Optional[List[float]]:
        """
        Finds the scale and offset values valid for a specific timestamp.

        Args:
            time (float): The timestamp.
        """
        if self.scale_offset_list is None:
            return None
        
        if len(self.scale_offset_list) == 1:
            dp : DataPoint = self.scale_offset_list[0]
            return dp.data.tolist() if dp else None

        times = np.array([dp.time for dp in self.scale_offset_list if dp])
        if times.size == 0: return None

        idx = np.searchsorted(times, time, side='right') - 1
        idx = np.clip(idx, 0, len(self.scale_offset_list) - 1)
        
        selected_dp : DataPoint = self.scale_offset_list[idx]
        return selected_dp.data.tolist() if selected_dp else None

    def is_match(self, dt: 'DataTrend') -> bool:
        """Checks if this model matches the operational state of another trend."""
        if dt is not None:
            _state = dt.get_state()
            return _state is not None and _state == self.get_state()
        else:
            return False

    @property
    def is_disjoint(self) -> bool:
        """Returns True if the current state is marked as disjoint."""
        if self.state is None:
            return False
        else:
            flag = self.state.flag
            return flag is not None and flag == DISJOINT

    def get_algorithm_data(self) -> Optional[AlgorithmData]:
        """Returns a picklable data representation of this trend for storage/IPC."""
        if self.params is None:
            return None
        else:
            return SingleStateData(
                mnemonic_id= self.mnemonic_id,
                alg_name=self.alg_name,
                pattern_period=self.pattern_period,
                pattern_times = self.pattern_times,
                ref_time = self.ref_time,
                stat_list=self.stat_list,
                scale_offset_list=self.scale_offset_list if self.scale_offset_list else None,
                sigma=self.sigma,
                tpc=self.tpc,
                params=self.params,
                state= self.state.name if self.state else DEFAULT
            )

    def set_algorithm_data(self, algorithm_data: AlgorithmData):
        """
        Populates this model from an AlgorithmData object.

        Args:
            algorithm_data (AlgorithmData): The data source.
        """
        self.alg_name = algorithm_data.alg_name
        self.mnemonic_id = algorithm_data.mnemonic_id
        single_state_data : SingleStateData = algorithm_data
        self.set_data_model_time(single_state_data.pattern_times, single_state_data.pattern_period)
        self.ref_time = single_state_data.ref_time
        self.stat_list = single_state_data.stat_list
        self.scale_offset_list = single_state_data.scale_offset_list
        self.sigma = single_state_data.sigma
        self.tpc = single_state_data.tpc
        self.set_params(single_state_data.params)

    def __getitem__(self, item):
        if item in DataTrend.TREND_KEYS:
            return getattr(self, item)
        else:
            raise KeyError(f"Key '{item}' not defined in DataTrend.")

    def __setitem__(self, key, value):
        if key in DataTrend.TREND_KEYS:
            setattr(self, key, value)
            return
        else:
            raise KeyError(f"Key '{key}' not defined in DataTrend.")

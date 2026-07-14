import logging
import os
import sys
import traceback
from asyncio import sleep
from typing import List, Optional, Tuple

import numpy as np
from influxdb_client.client.flux_table import TableList

import plugin_manager
from algorithm.algorithm_data import AlgorithmData, TREND
from algorithm.algorithm_def import AlgorithmDef
from algorithm.hybrid_state_data import HybridStateData
from algorithm.single_state_data import SingleStateData
from config import sdt_constants, sdt_config
from config.sdt_constants import SHORTTERM, DEFAULT
from dataio.influxdb.event_io import EventIO
from dataio.influxdb.influxdb_util import MODELIO
from posttraining.clustering.sdt_event_data import SDTEventData
from posttraining.mnemonic_status import MnemonicStatus
from sdt_exception import SDTException
from sdtdb import sdt_db
from training.training_worker import HYBRID

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataio.sdt_data_io import SDTDataTrainingIO
from algorithm.data_point import DataPoint
from dataio.influxdb import influx_script_generator as script_gen, influxdb_util
from training import data_buffer as db, data_buffer
from util import time_util as tu

from influxdb_client import Point
from influxdb_client.client.write_api import WriteOptions
from influxdb_client.client.exceptions import InfluxDBError

# --- Constants ---
CONTEXT = "InfluxDBDataIO"
AIMSINPUT = "aimsinput" # Placeholder for mode
OPSTATUS = "opstatus"   # Placeholder for mode


def _get_stat_array(stat_array: List[DataPoint], time: float, ref_time: float, pattern_period: float, state: str) -> Optional[np.ndarray]:
    if state == DEFAULT:
        _index =int (round((time - ref_time) / pattern_period))
        if len(stat_array) > _index >= 0:
            if stat_array[_index] is None:
                stat_array[_index] = DataPoint(ref_time, np.zeros(4))
            return stat_array[_index].data
        else:
            return None
    else:
        for index in range(len(stat_array)):
            if stat_array[index] is None:
                stat_array[index] = DataPoint(ref_time, np.zeros(4))
                return stat_array[index].data
            else: # This else branch is problematic. It should be an elif or part of the if.
                if time == stat_array[index].time:
                    return stat_array[index].data
    return None


def create_algorithm_data(mnemonic_id: str, state: str) -> Optional[AlgorithmData]:
    algorithm = AlgorithmDef(sdt_db.get_algorithm(mnemonic_id))
    algorithm_name = algorithm.get_name()

    state_type = algorithm.get_state(state)
    if state_type is not None:
        algorithm_name = state_type.algorithm
    return SingleStateData(
        mnemonic_id=mnemonic_id,
        alg_name= algorithm_name,
        ref_time=0,
        pattern_period=0,
        pattern_times=None,
        stat_list= None,
        scale_offset_list=None,
        sigma=0.0,
        tpc=1.0,
        params=None,
        num_pattern_in_training=1,
        state=state
        )


def get_data_trend_from_record(table_list : TableList, mnemonic_id: str, state : str)-> Optional[AlgorithmData]:
    algorithm_data = create_algorithm_data(mnemonic_id, state)
    ref_time = 0
    param_dim, model_param_dim = get_param_number(algorithm_data)
    params = [0]*model_param_dim
    for table in table_list:
        for record in table.records:
            ref_time =int (record.get_time().timestamp())
            algorithm_data.ref_time = ref_time
            field_name = record.get_field()
            if field_name and field_name == TREND:
                index = int(record['index'])
                if len(params) > index >= 0:
                    params[index] = record.get_value()
        if any(params):
            algorithm_data.sigma = params[0]
            algorithm_data.param = params[1:param_dim+1]
            if param_dim +1 < len(params):
                pattern_times = np.zeros(model_param_dim-param_dim)
                pattern_times[0] = ref_time
                pattern_times[1:] = params[param_dim+1:]
                algorithm_data.pattern_times = np.array(pattern_times)
    return algorithm_data

def get_param_number(algorithm_data: AlgorithmData)-> Tuple[int, int]:
    from algorithm.data_trend import DataTrend
    data_trend : DataTrend = plugin_manager.get_data_trend(algorithm_data.alg_name, algorithm_data.mnemonic_id)
    if data_trend:
        return data_trend.get_param_dim(), data_trend.get_model_param_dim()
    else:
        return 0, 0


def get_data_point(sigma_list : List[DataPoint], time : float, dim : int) -> Optional[DataPoint]:
    if not sigma_list:
        data = np.zeros(dim)
        data_point = DataPoint(time, data)
        sigma_list.append(data_point)
        return data_point
    for data_point in sigma_list:
        if data_point.time == time:
            return data_point
    data_point = DataPoint(time, np.zeros(dim))
    sigma_list.append(data_point)
    return data_point


class InfluxDBSDTDataIO(SDTDataTrainingIO):
    """
    InfluxDB implementation for reading and writing data training archives.
    """

    def __init__(self):
        self.query_api=None
        self.write_api=None
        self.client=None
        self.sdt_bucket=None
        self.ops_bucket=None
        self.org=None
        self.event_io : EventIO = None
        self._init_client_()

    def _init_client_(self):
        try:
            self.client = influxdb_util.get_influx_client(MODELIO)

            # Using SYNCHRONOUS write for simplicity and immediate feedback.
            # For high-throughput, ASYNCHRONOUS with larger batch sizes is better.
            self.sdt_bucket = influxdb_util.get_influx_bucket(influxdb_util.MODELIO)
            self.ops_bucket = influxdb_util.get_influx_bucket(influxdb_util.OPSTATUS)
            self.org = influxdb_util.get_influx_org(influxdb_util.MODELIO)
            logging.info(f"{CONTEXT}: InfluxDB client initialized for org '{self.org}'.")
        except InfluxDBError as e:
            raise SDTException(f"Failed to initialize InfluxDB client: {e}")

    def init_query_api(self):
        try:
            self.query_api = self.client.query_api()
        except InfluxDBError as e:
            logging.error(f"{CONTEXT}: Failed to initialize query API {e}")
            return None

    def init_write_api(self):
        try:
            options = WriteOptions(
                batch_size=500,
                flush_interval=3_000,
                jitter_interval=2_000,
                retry_interval=5_000,
                max_retries=5,
                max_retry_delay=30_000,
                max_close_wait=10_000,
                exponential_base=2
            )
            self.write_api = self.client.write_api(option=options)
        except InfluxDBError as e:
            logging.error(f"{CONTEXT}: Failed to initialize write API {e}")
            return None

    def write_data_trend(self, algorithm_data_list :List[AlgorithmData]):
        """
        Writes the data trends from a node to InfluxDB.
        """
        if not algorithm_data_list:
            return
        mnemonic_id = algorithm_data_list[0].mnemonic_id
        ##logging.info(f"{CONTEXT}: Writing {mnemonic_id} trend to InfluxDB")

        try:
            # Using line protocol for flexibility as generated by the script generator
            trend_lines = []
            for algorithm_data in algorithm_data_list:
                if algorithm_data.alg_name == HYBRID:
                    h_trend : HybridStateData = algorithm_data
                    sub_trends = h_trend.data_trend_list
                    for sub_trend in sub_trends:
                        trend_lines.extend(script_gen.get_trend_points_for_line_protocol(sub_trend))
                        # ops_lines.extend(script_gen.get_ops_trend_outputs(trend)) # Placeholder for complex OPS points
                else:
                    trend_lines.extend(script_gen.get_trend_points_for_line_protocol(algorithm_data))
            if trend_lines:
                self._write_records(bucket=self.sdt_bucket, lines=trend_lines)
            model_lines = script_gen.get_model_points_for_line_protocol(algorithm_data_list)
            if model_lines:
                self._write_records(bucket=self.ops_bucket, lines=model_lines)
        except Exception as e:
            error_details = traceback.format_exc()
            logging.error(f"{CONTEXT}: Error writing data to InfluxDB for {mnemonic_id}: {e}\n{error_details}")
            raise SDTException(f"Error writing data to InfluxDB for {mnemonic_id}: {e}")

    def _write_records(self, bucket: str, lines: List[Point]):
        """
        Writes a list of line protocol Points to a specified bucket.
        """
        if not lines:
            return
            
        # Filter out None and empty strings
        valid_lines = [line for line in lines if line]
        
        if not valid_lines:
            return

        try:
            if not self.client:
                self._init_client_()
            if self.write_api is None:
                self.init_write_api()
            self.write_api.write(bucket=bucket, org=self.org, record=lines)
            self.write_api.flush()
        except InfluxDBError as e:
            logging.error(f"{CONTEXT}: Failed to write to bucket '{bucket}': {e}")
            # In a real scenario, you might add retry logic here.

    def get_data_trend(self, mnemonic_id: str, start: float, end: float, state: str) -> Optional[AlgorithmData]:
        # This is a complex method. A full implementation requires careful handling of Flux results.
        # This placeholder demonstrates the basic query and parsing flow.
        logging.info(f"{CONTEXT}: Retrieving DataTrend for {mnemonic_id} from {start} to {end}")

        if data_buffer.session_type == SHORTTERM:
            return self.get_short_term_trend(mnemonic_id, start, end, state)
        else:
            return self.get_lt_term_trend(mnemonic_id, start, end)

    def get_lt_term_trend(self, mnemonic_id: str, start : float, end : float) -> Optional[List[AlgorithmData]]:
        if not self.client:
            self._init_client_()
        if self.query_api is None:
            self.init_query_api()
        if self.query_api:
            input_trends : List[AlgorithmData]=[]
            for stat_string in sdt_constants.MNPOSTFIX:
                query = script_gen.get_flux_query_string(mnemonic_id, start, end, "lttrend", None, stat_string)
                table_list = self.query_api.query(query, self.org)
                input_trends[sdt_constants.get_stat_index(stat_string)] = self.get_long_term_trend(table_list, mnemonic_id, stat_string)
                logging.info(f"{CONTEXT} Retrieve long term data trend for {mnemonic_id} with postfix {stat_string}")
            return input_trends
        else:
            logging.error(f"{CONTEXT}: Query API not available.")
            return None

    def get_long_term_trend(self, table_list : TableList, mnemonic_id: str, post_fix: str) -> Optional[AlgorithmData]:
        return None

    def get_short_term_trend(self, mnemonic_id: str, start: float, end: float, state: str) -> Optional[AlgorithmData]:
        if not self.client:
            self._init_client_()
        if self.query_api is None:
            self.init_query_api()
        logging.info(f"{CONTEXT}: get data trend for {mnemonic_id} from {tu.get_time_tag_from_seconds(start)} to {tu.get_time_tag_from_seconds(end)}")
        query = script_gen.get_flux_query_string(mnemonic_id, start, end, "trend", state)
        table_list = None
        #logging.info(f"{CONTEXT}: Flux query for the data trend retrieval: {query}")
        if query:
            try:
                table_list = self.query_api.query(query=query)
            except InfluxDBError as e:
                logging.error(f"{CONTEXT}: Failed to execute query {e}")
                table_list = self.query_api.query(query, self.org)
        if table_list and len(table_list) > 0:
            return get_data_trend_from_record(table_list, mnemonic_id, state)
        return None

    def _retrieve_stats(self, input_trend: SingleStateData) -> Optional[List[DataPoint]]:
        if not self.client:
            self._init_client_()
        if self.query_api is None:
            self.init_query_api()

        algorithm  = AlgorithmDef(sdt_db.get_algorithm(input_trend.mnemonic_id))
        start_s = 0
        end_s = 0
        num_patterns = sdt_config.num_pattern_in_training
        if input_trend.state == DEFAULT:
            start_s = input_trend.ref_time - 300
            if algorithm.is_disjoint(DEFAULT):
                end_s = input_trend.ref_time + 300
            else:
                if algorithm.np > 0:
                    num_patterns = algorithm.np
                pattern_period = input_trend.pattern_period
                if pattern_period == 0:
                    pattern_period = db.get_default_model_time(input_trend.mnemonic_id).model_period
                end_s = input_trend.ref_time + 2 * pattern_period*num_patterns - 150
        else:
            pattern_times  = input_trend.pattern_times
            if pattern_times != None and len(pattern_times) > 1:
                start_s = pattern_times[0] - 600
                end_s = pattern_times[-1]
            else:
                start_s =0
                end_s = 0
        logging.debug(f"{CONTEXT} Retrieve Stat Data for {input_trend.mnemonic_id} from {tu.get_time_tag_from_seconds(start_s)} to {tu.get_time_tag_from_seconds(end_s)}")
        if start_s > 0 and end_s > 0:
            stat_query = script_gen.get_flux_query_string(input_trend.mnemonic_id, start_s, end_s, "stat", input_trend.state)
            table_list  = None
            dim = 2*num_patterns
            if algorithm.is_disjoint(input_trend.state):
                dim = 1
            stat_array = []
            try:
                table_list = self.query_api.query(stat_query, self.org)
            except InfluxDBError as e:
                logging.error(f"{CONTEXT}: Failed to execute query: {e}")
                sleep(200)
                table_list = self.query_api.query(stat_query, self.org)
            if table_list and len(table_list) > 0:
                for table in table_list:
                    time = 0
                    data = np.zeros(4)
                    for record in table.records:
                        time =int (record.get_time().timestamp())
                        data = _get_stat_array(stat_array, time, input_trend.ref_time, input_trend.pattern_period, input_trend.state)
                        if data :
                            index_string = record["index"]
                            index = int(index_string)
                            if len(data) > index > 0:
                                data[index] = record.get_value()
                    stat_array.append(DataPoint(time=time, data=data))
            if stat_array:
                return stat_array
            else:
                return None
            
            # In a real implementation, you would parse the FluxTables to reconstruct the DataTrend object.
            # This is non-trivial as it involves collecting all parameters for a given timestamp.
            # For this conversion, we'll return a placeholder.
            
            # Placeholder logic:
            # 1. Group records by time.
            # 2. For each time, create a DataTrend object.
            # 3. Populate the 'params' array from the records.
        return None

    # --- Other abstract methods would be implemented here ---
    def write_status(self, status_list: List[MnemonicStatus]):
        if not status_list:
            logging.warning(f"{CONTEXT}: No status to write.")
            return
        if not self.client:
            self._init_client_()
        if not self.write_api:
            self.init_write_api()
        logging.info(f"{CONTEXT}: Writing OPS status to InfluxDB")
        for status in status_list:
            point = script_gen.get_status_points_for_line_protocol(status)
            if point:
                self.write_api.write(bucket=self.ops_bucket, org=self.org, record=point)

    def get_data_stats(self, mnemonic_id: str, start: float, end: float, state: str) -> Optional[List[List[DataPoint]]]:
        logging.warning(f"{CONTEXT}: get_data_stats is not implemented.")
        return None

    def get_sigma(self, mnemonic_id, start : float, end : float, state : str) -> Optional[List[DataPoint]]:
        query_str = script_gen.get_sigma_query_string(mnemonic_id, start, end, state)
        try:
            if not self.client:
                self._init_client_()
            if self.query_api is None:
                self.init_query_api()
            table_list = self.query_api.query(query_str, self.org)
        except InfluxDBError as e:
            logging.error(f"{CONTEXT}: Failed to execute query: {e}")
            return None
        sigma_list = []
        dim = 1
        index = 0
        if table_list and len(table_list) > 0:
            for table in table_list:
                for record in table.records:
                    time = int(record.get_time().timestamp())
                    data_point = get_data_point(sigma_list, time, dim)
                    data_point.data[index] = record.get_value()
        return sigma_list

    def get_model_params(self, mnemonic_id, start: float, end: float, state: str) -> Optional[List[DataPoint]]:
        logging.warning(f"{CONTEXT}: get_model_params is not implemented.")
        return None

    def get_event_history(self, start_time: float, end_time: float) -> Optional[List[SDTEventData]]:
        if self.event_io is None:
            self.event_io = EventIO()
        event_list= self.event_io.get_event_history(start_time, end_time)
        return event_list

    def write_events(self, event_list: List[SDTEventData], cutoff: float):
        if self.event_io is None:
            self.event_io = EventIO()
        self.event_io.write_events(event_list, cutoff)

    def close(self):
        """
        Closes the InfluxDB client and flushes any pending writes.
        """
        logging.info(f"{CONTEXT}: Closing InfluxDB client.")
        if self.write_api:
            try:
                self.write_api.flush()
                self.write_api.close()
                self.write_api=None
            except Exception:
                pass
        if self.query_api:
            self.query_api=None
        if self.client:
            try:
                self.client.close()
                self.client=None
            except Exception:
                pass
        if self.event_io:
            self.event_io.close()
        logging.info(f"{CONTEXT}: InfluxDB client closed.")

    def get_lt_trend(self, mnemonic_id, start: float, end: float) -> Optional[List[AlgorithmData]]:
        return None
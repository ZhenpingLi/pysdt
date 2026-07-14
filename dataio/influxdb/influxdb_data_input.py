import gc
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
from influxdb_client.client.exceptions import InfluxDBError

from config import sdt_config
from dataio.influxdb import influxdb_util
# Use absolute imports from the project root, assuming the project is run as a package.
from dataio.sdt_data_input import SDTDataInput, _merge_data_list, commands
from dataplot.data_plot import HandleDataPlot
from sdtdb import sdt_db
from training.training_set import TrainingSet
from util import time_util

# --- Main Class Definition ---
CONTEXT = "DATAINPUT"




class InfluxDBDataInput(SDTDataInput):
    def __init__(self):
        self.query_api = None
        self.client = None
        self._init_client_()

    def _init_client_(self):
        self.client = influxdb_util.get_influx_client(influxdb_util.DATAINPUT)
        self.bucket_name = influxdb_util.get_influx_bucket(influxdb_util.DATAINPUT)
        self.org = influxdb_util.get_influx_org(influxdb_util.DATAINPUT)
        self.agg_window = sdt_config.get_int_property("AGGREGATEWINDOW")
        self.query_api = None

    def get_data(self, mnemonic_id: str, start: float, end: float) -> Optional[TrainingSet]:
        """
        Main method to retrieve and process data for a given logic path.
        It fetches data for all related mnemonics and transforms it into a
        single, row-oriented list of DataPoint objects, using interpolation
        to fill missing values at common timestamps.
        """

        mn_names = sdt_db.get_mnemonic_names(mnemonic_id)
        if not mn_names:
            return None

        # --- 1. Fetch data for all mnemonics (column-oriented) ---
        input_data, output_data = self._process_sync_query(mn_names[0], start, end)
        if not input_data or not output_data:
            return None
        training_set = TrainingSet(
            mnemonic_id=mnemonic_id,
            inputs=input_data,
            raw=output_data,
            outputs=output_data,
            dqf=np.ones(len(input_data), dtype=np.int8)
        )
        column_times = []
        column_data = []
        if len(mn_names)>1:
            for name in mn_names[1:]:
                _times, _data = self._process_sync_query(name, start, end)
                column_times.append(_times)
                column_data.append(_data)
            _merge_data_list(training_set, column_times, column_data)

        # --- 2. Collect all unique timestamps across all mnemonics ---
        _function = sdt_db.get_algorithm_attribute(mnemonic_id, "function")
        if _function:
            _function = _function.strip() # Clean up potential whitespace
            if _function in commands:
                if training_set.outputs:
                    commands[_function](training_set)
            else:
                logging.warning(f"{CONTEXT}: Function '{_function}' not found in commands list.")
                
        return training_set

    def _process_sync_query(self, mnemonic_id: str, start : float, end : float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds and executes a Flux query, returning the data as a list of (time, value) tuples.
        """
        flux_string = self.get_flux_query_string(start, end, mnemonic_id, False)
        if not flux_string:
            return None

        #logging.info(f"{CONTEXT} Executing Flux Query:\n{flux_string}\n")
        
        try:
            if not self.client:
                self._init_client_()
            if not self.query_api:
                self.query_api = self.client.query_api()
            tables = self.query_api.query(org=self.org, query=flux_string)
        except InfluxDBError as ex:
            logging.error(f"{CONTEXT} Error querying InfluxDB: {ex}")
            return []
        logging.info(f"{CONTEXT} retrieved the data for {mnemonic_id} from {time_util.get_time_tag_from_seconds(start)} to {time_util.get_time_tag_from_seconds(end)}")
        times = None
        raw_data = None
        if tables:
            for table in tables:
                if not table.records:
                    continue
                # Iterate and create DataPoints
                times = np.array([float(record.get_time().timestamp()) for record in table.records], dtype=np.float64)
                raw_data = np.array([float(record.get_value()) for record in table.records], dtype=np.float32)

                table.records.clear()
            
            # Clear tables list to release references
            tables.clear()
        gc.collect()
        return times.reshape(-1,1), raw_data

    def get_flux_query_string(self, start: float, end: float, mnemonic_id: str, append_pd: bool) -> str:
        """
        Translates Java InfluxDB query builder to Python 3.14+
        """
        start_s = datetime.fromtimestamp(start, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        end_s = datetime.fromtimestamp(end, tz=timezone.utc).isoformat().replace("+00:00", "Z")

        mn_name = mnemonic_id
        agg_window = 2
        agg_string = sdt_db.get_mnemonic_attribute(mnemonic_id, "aggwindow")
        if agg_string:
            agg_window = int(agg_string)

        query_parts = [
            f'from(bucket: "{self.bucket_name}")',
            f' |> range(start: {start_s}, stop: {end_s})',
            f' |> filter(fn: (r) => (r["_measurement"] == "{mn_name}"))'
        ]

        if agg_window > 0:
            query_parts.append(f' |> aggregateWindow(every: {agg_window}s, fn: mean, createEmpty: false)')
            query_parts.append(' |> yield(name: "mean")')
        if append_pd :
            query_parts.append(' |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")')

        return "".join(query_parts)

    def close(self):
        if self.client:
            try:
                if self.query_api:
                    self.query_api=None
                self.client.close()
                self.client=None
                logging.info(f"{CONTEXT}: InfluxDB data input client is closed.")
            except Exception:
                pass # Ignore errors during close

if __name__ == '__main__':
    # Example Usage
    influx_input = None
    try:
        import config.sdt_config
        sdt_config.set_sat_id("g19")
        sdt_config.load_config()
        sdt_db.init_database()
        import plugin_manager as pm
        influx_input = pm.get_sdt_data_input("default")
        start_time= time_util.get_time_from_string("2025/199")
        end_time = time_util.get_time_from_string("2025/201")
        lg_path = ["ABI", "FPM_temp"]
        logging.info(f"{CONTEXT} start data retrieval for {lg_path}")
        data = influx_input.get_data(lg_path, start_time, end_time)
        logging.info(f"{CONTEXT} end data retrieval for {lg_path}")
        if data:
            logging.info(f"{CONTEXT} Successfully retrieved and interpolated {len(data)} DataPoint objects.")
            lg_path_i = lg_path + ["1"]
            HandleDataPlot.plot_data_trend(data, lg_path_i)
        else:
            print("No data retrieved.")
    except Exception as e:
        logging.exception(f"An error occurred in the main execution block: {e}")
    finally:
        if influx_input:
            influx_input.close()
            print("InfluxDB client closed.")

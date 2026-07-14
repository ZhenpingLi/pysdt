import logging
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

from influxdb_client.client.exceptions import InfluxDBError

from posttraining.clustering.sdt_event_data import SDTEventData
from sdt_exception import SDTException

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config.sdt_config as sdt_config
from dataio.influxdb import influxdb_util
from sdtdb import sdt_db
# Assuming these classes will be moved to their own files or are importable


# NOTE: This implementation requires the InfluxDB client library.

from influxdb_client import Point
from influxdb_client.client.write_api import WriteOptions

CONTEXT = "EventIO"
EVENTHISTORY = "eventhistory"
AIMSINPUT = 0 # Corresponds to InfluxDBUtil.DATAINPUT (or should it be MODELIO? Java uses AIMSINPUT which maps to DATAINPUT in util)

def get_event_string(event_data_list: List[SDTEventData]) -> List[Point]:
    """
    Converts an OutlierEvent into a list of InfluxDB line protocol strings.
    """
    ingest_list = []
    for event_data in event_data_list:
        mnemonic_id = event_data.name
        event_value = event_data.event_value
        dt = datetime.fromtimestamp(event_data.event_start, tz=timezone.utc)
        period = event_data.event_end - event_data.event_start
        event_state = event_data.event_state
        event_point = Point(EVENTHISTORY).field("value", f"{mnemonic_id}|{event_value:.2f}|{period:.2f}|{event_state}").time(dt)
        ingest_list.append(event_point)

    return ingest_list


def get_event_data(field_value: str, ref_time: float) -> Optional[SDTEventData]:
    """
    Parses a stored event string back into an OutlierCluster object.
    """
    tokens = field_value.split('|')
    if len(tokens) < 3:
        return None

    if sdt_db.exist(tokens[0]): # Check if ID exists
        # We need to create a dummy Outlier to initialize OutlierCluster
        # or modify OutlierCluster to accept direct initialization.
        # Assuming OutlierCluster can be initialized with path and time.
        # Since OutlierCluster in event_clustering.py takes an Outlier, we mock one.
        #from algorithm.outlier import Outlier

        return SDTEventData(
            name=tokens[0],
            event_start=ref_time,
            event_end=ref_time+float(tokens[2]),
            event_value=float(tokens[1]),
            event_type="outlier",
            event_state=tokens[3]
        )
    else:
        return None


class EventIO:
    """
    Handles reading and writing event data to/from the archive (InfluxDB).
    """

    def __init__(self):
        self.bucket_name = influxdb_util.get_influx_bucket(AIMSINPUT)
        self.client = influxdb_util.get_influx_client(AIMSINPUT)
        self.org = influxdb_util.get_influx_org(AIMSINPUT)
        self.oc_join_time: float = float(sdt_config.get_config_value("OCJOINTIME") or 5.0)
        self.SCALE: float = float(sdt_config.get_config_value("OUTLIERSCALE") or 100.0)
        self.write_api = None
        self.query_api = None
        
        self.event_period_limit = float(sdt_config.get_config_value("EVENTTIMELIMIT") or 3600.0)

    def write_events(self, event_list: List[SDTEventData], cutoff: float):
        """
        Writes a list of outlier events to the archive.
        """
        event_ingest_list: List[Point] = []
        
        if event_list:
            filtered_event_list = [event for event in event_list if event.event_start > cutoff]
            if filtered_event_list:
                event_ingest_list = get_event_string(filtered_event_list)
        else:
            logging.info(f"{CONTEXT}: No event for the current training session")

        # Debug logging to file (skipped for brevity, can be added if needed)

        try:
            if self.write_api is None and self.client:
                self.write_api = self.client.write_api(write_options=WriteOptions(batch_size=500))
            
            if event_ingest_list and self.write_api:
                self.write_api.write(bucket=self.bucket_name, org=self.org, record=event_ingest_list)
                
        except Exception as ex:
            logging.error(f"{CONTEXT}: Error in writing events into influxdb: {ex}")
            raise SDTException(f"Error writing events: {ex}")

    def get_event_history(self, start_time: float, end_time: float) -> List[SDTEventData]:
        """
        Retrieves event history from the archive.
        """
        event_data_list: List[SDTEventData] = []
        
        if self.query_api is None and self.client:
            self.query_api = self.client.query_api()
            
        try:
            query_string = self.get_event_query_string(start_time, end_time, "value")
            if self.query_api:
                tables = self.query_api.query(query_string, org=self.org)
                
                for table in tables:
                    for record in table.records:
                        ref_time = 0.0
                        if record.get_time():
                            ref_time = record.get_time().timestamp()
                        
                        field_value = record.get_value()
                        if field_value:
                            event_data = get_event_data(field_value, ref_time)
                            if event_data is not None:
                                event_data_list.append(event_data)
                                
        except Exception as ex:
            logging.error(f"{CONTEXT}: Error in retrieving the event history files: {ex}")
            raise SDTException(f"Error retrieving event history: {ex}")

        return event_data_list

    def get_event_query_string(self, start: float, end: float, field: str) -> str:
        """
        Constructs the Flux query string.
        """
        start_s = datetime.fromtimestamp(start, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        end_s = datetime.fromtimestamp(end, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        
        flux_query = (
            f'from(bucket: "{self.bucket_name}")\n'
            f' |> range(start: {start_s}, stop: {end_s})\n'
            f' |> filter(fn: (r) => (r._measurement == "{EVENTHISTORY}" and r._field == "{field}"))'
        )
        return flux_query

    def close(self):
        """
        Closes the InfluxDB client.
        """
        if self.write_api:
            try:
                self.write_api.close()
            except InfluxDBError:
                logging.error(f"Error in close the write API", exc_info=True)
        if self.client:
            try:
                self.client.close()
            except InfluxDBError:
                logging.error(f"Error in close the client", exc_info=True)
        self.write_api = None
        self.query_api = None
        self.client = None


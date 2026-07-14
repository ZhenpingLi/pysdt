from typing import List, Dict

import numpy as np

from dataio.sqlite.tlm_ingest.sqlite_data_ingest_thread import SQLiteDataIngestThread
from dataio.sqlite.tlm_ingest.sqlite_ingest_packet import SQLiteIngestPacket
from dataio.sqlite.tlm_ingest.tlm_data import TlmData


class TlmSQLiteDataConverter:
    """
    Manages the conversion of raw telemetry data (TlmData) into SQLiteIngestPacket
    objects and orchestrates their ingestion into the SQLite database via 
    SQLiteDataIngestThread.
    
    This class handles data compression, batching, and interpolation to prepare 
    telemetry for efficient storage.
    """
    def __init__(self, thread: SQLiteDataIngestThread):
        """
        Initializes the TlmSQLiteDataConverter.

        Args:
            thread (SQLiteDataIngestThread): The thread responsible for writing 
                data to the SQLite database.
        """
        self.write_thread = thread
        self.tlm_packet_dict: Dict[str, SQLiteIngestPacket] = {}

    def ingest_data(self):
        """
        Triggers the ingestion of accumulated telemetry packets into the SQLite database.
        
        This method converts the collected TlmData into SQLiteIngestPacket objects, 
        passes them to the write thread, and then clears the internal buffer.
        """
        if self.tlm_packet_dict:
            sqlite_packet_list: List[SQLiteIngestPacket] = self.get_sqlite_packet_list()
            self.write_thread.ingest_data(sqlite_packet_list)
            self.tlm_packet_dict.clear()

    def convert_to_sql_packet(self, tlm_list: List[TlmData]):
        """
        Accumulates raw telemetry data points (TlmData) into internal packets.
        
        For each TlmData object, it either appends the data to an existing 
        SQLiteIngestPacket for that mnemonic or creates a new packet.

        Args:
            tlm_list (List[TlmData]): A list of raw telemetry data points.
        """
        for tlm_data in tlm_list:
            mnemonic_id = tlm_data.name
            if mnemonic_id in self.tlm_packet_dict:
                packet = self.tlm_packet_dict[mnemonic_id]
                packet.values.append(tlm_data.value)
                packet.times.append(tlm_data.timestamp)
            else:
                # Assuming a default frequency for initial packet creation
                packet = SQLiteIngestPacket(mnemonic_id=mnemonic_id, times=[tlm_data.timestamp], frequency=1000, values=[tlm_data.value])
                self.tlm_packet_dict[mnemonic_id] = packet

    def get_sqlite_packet_list(self) -> List[SQLiteIngestPacket]:
        """
        Processes the accumulated telemetry packets, performing interpolation 
        and frequency adjustment, and returns them as a list of SQLiteIngestPacket.

        Returns:
            List[SQLiteIngestPacket]: A list of prepared packets ready for ingestion.
        """
        sqlite_packet_list: List[SQLiteIngestPacket] = []
        for packet in self.tlm_packet_dict.values():
            num_samples = len(packet.times)
            
            # Calculate effective frequency based on actual data spread
            if num_samples > 1:
                frequency = round((packet.times[-1] - packet.times[0]) / 1000 / num_samples)
                if frequency < 1:
                    frequency = 1
            else:
                frequency = 1 # Default frequency for single-point packets

            # Interpolate data to a regular grid based on calculated frequency
            if num_samples > 1:
                numb_entries = (packet.times[-1] - packet.times[0]) / 1000 / frequency
                filtered_times = packet.times[0] + (np.arange(numb_entries) * 1000 * frequency)
                filtered_values = np.interp(filtered_times, np.array(packet.times), np.array(packet.values)).tolist()
            else: # Handle single data point case
                filtered_times = np.array(packet.times)
                filtered_values = packet.values

            new_packet = SQLiteIngestPacket(mnemonic_id=packet.mnemonic_id, times=filtered_times.tolist(), frequency=frequency, values=filtered_values)
            sqlite_packet_list.append(new_packet)
            
        return sqlite_packet_list

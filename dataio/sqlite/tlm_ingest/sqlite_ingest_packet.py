from dataclasses import dataclass
from typing import List


@dataclass
class SQLiteIngestPacket:
    """
    Data structure representing a packet of telemetry data for SQLite ingestion.
    
    This dataclass encapsulates a batch of data points for a single mnemonic, 
    including their timestamps and values. It is used to transfer data between 
    the retrieval process and the SQLite ingestion thread.

    Attributes:
        mnemonic_id (str): The unique identifier for the telemetry mnemonic.
        times (List[int]): A list of absolute timestamps (typically in 
            milliseconds) for each data point in the packet.
        frequency (float): The sampling frequency of the telemetry data.
        values (List[float]): A list of telemetry data values corresponding to 
            the timestamps in `times`.
    """
    mnemonic_id : str
    times: List[int]
    frequency : float
    values : List[float]

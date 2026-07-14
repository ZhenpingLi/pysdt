from dataclasses import dataclass


@dataclass
class TlmData:
    """
    Represents a single telemetry data point formatted for ingestion into an archive.

    This dataclass provides a structured representation of telemetry data, including
    its subsystem, mnemonic name, timestamp, and value.

    Attributes:
        subsystem (str): The name of the satellite subsystem (e.g., 'EPS', 'GNC').
        mn_name (str): The mnemonic name of the telemetry point (e.g., 'BAT_VOLT').
        time (float): The absolute timestamp of the data point in seconds (Unix epoch).
        value (float): The telemetry value.
    """
    subsystem: str
    mn_name: str
    time: float = 0.0
    value: float = 0.0

    @property
    def timestamp(self) -> int:
        """
        Returns the timestamp of the data point in milliseconds.

        Returns:
            int: The timestamp rounded to the nearest millisecond.
        """
        return round(self.time * 1000)

    @property
    def name(self) -> str:
        """
        Alias for the mnemonic name.

        Returns:
            str: The mnemonic name.
        """
        return self.mn_name

    @property
    def system(self) -> str:
        """
        Alias for the subsystem name.

        Returns:
            str: The subsystem name.
        """
        return self.subsystem

    def __str__(self) -> str:
        """
        Returns a formatted string representation suitable for logging or InfluxDB line protocol.

        Returns:
            str: A string in the format 'subsystem,name=mn_name value=value timestamp_ms'.
        """
        time_long = round(self.time * 1000)
        return f"{self.subsystem},name={self.mn_name} value={self.value} {time_long}"


import math
import re
from datetime import datetime, timedelta, timezone

# --- Constants ---
DAY_IN_SECONDS = 86400
"""Number of seconds in a day."""

HOUR_IN_SECONDS = 3600
"""Number of seconds in an hour."""

TIMEJ2000 = 946728000.0
"""Timestamp offset for J2000 epoch (2000-01-01T12:00:00Z)."""

TIMEJ1958 = -378691200.0
"""Timestamp offset for 1958 epoch."""

ONEHOUR = 3600000
"""One hour in milliseconds."""

WEEK = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
"""Day of the week labels."""

TIMEPATTERN = [
    r".",
    r"^\d",
    r"^\d{2}",
    r"^\d{3}",
    r"^\d{4}",
    r"^\d{4}/",
    r"^\d{4}/\d",
    r"^\d{4}/\d{2}",
    r"^\d{4}/\d{3}",
    r"^\d{4}/\d{3}/",
    r"^\d{4}/\d{3}/\d",
    r"^\d{4}/\d{3}/\d{2}",
    r"^\d{4}/\d{3}/\d{2}:",
    r"^\d{4}/\d{3}/\d{2}:\d",
    r"^\d{4}/\d{3}/\d{2}:\d{2}",
    r"^\d{4}/\d{3}/\d{2}:\d{2}:",
    r"^\d{4}/\d{3}/\d{2}:\d{2}:\d",
    r"^\d{4}/\d{3}/\d{2}:\d{2}:\d{2}",
    r"^\d{4}/\d{3}/\d{2}:\d{2}:\d{2}\.",
    r"^\d{4}/\d{3}/\d{2}:\d{2}:\d{2}\.\d",
    r"^\d{4}/\d{3}/\d{2}:\d{2}:\d{2}\.\d{2}",
    r"^\d{4}/\d{3}/\d{2}:\d{2}:\d{2}\.\d{3}"
]
"""Regex patterns for identifying partially completed time strings."""

def get_double_time_from_string(time_string: str) -> float:
    """
    Converts a standard ISO 8601 time string into Unix seconds.
    
    Expected format: YYYY-MM-DDTHH:MM:SSZ

    Args:
        time_string (str): The ISO 8601 formatted string.

    Returns:
        float: Seconds since the 1970 Unix epoch.
    """
    if time_string.endswith("Z"):
        time_string = time_string[:-1]
    
    try:
        dt = datetime.fromisoformat(time_string)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        raise ValueError(f"Could not parse ISO time string: {time_string}")

def get_time_from_string(s: str) -> float:
    """
    Converts an AIMS-style timestamp (YYYY/DDD/HH:MM:SS.mmm) into Unix seconds.
    
    DDD is the Day of Year (1-366). The string can be partially provided, 
    with missing components defaulting to 1970/001/00:00:00.

    Args:
        s (str): The AIMS time string.

    Returns:
        float: Seconds since the 1970 Unix epoch.
        
    Raises:
        ValueError: If the string does not match any recognized time patterns.
    """
    if not is_time_pattern(s):
        raise ValueError(f"Invalid time format: {s}")

    year, doy, hour, minute, second, microsecond = 1970, 1, 0, 0, 0, 0
    tokens = re.split(r'[/:.]', s)
    
    if len(tokens) >= 1: year = int(tokens[0])
    if len(tokens) >= 2: doy = int(tokens[1])
    if len(tokens) >= 3: hour = int(tokens[2])
    if len(tokens) >= 4: minute = int(tokens[3])
    if len(tokens) >= 5: second = int(tokens[4])
    if len(tokens) >= 6: 
        ms_str = tokens[5]
        if len(ms_str) > 3: ms_str = ms_str[:3]
        microsecond = int(ms_str) * 1000

    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
    dt = dt.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)
    
    return dt.timestamp()

def is_time_pattern(input_str: str) -> bool:
    """
    Validates if a string represents a valid partial or full AIMS time pattern.

    Args:
        input_str (str): The string to validate.

    Returns:
        bool: True if the string length matches a known pattern and the 
            regex matches.
    """
    length = len(input_str)
    if 0 < length < len(TIMEPATTERN):
        pattern = re.compile(TIMEPATTERN[length])
        return bool(pattern.match(input_str))
    return False

def get_datetime(time_sec: float) -> datetime:
    """
    Converts Unix seconds into a UTC datetime object.

    Args:
        time_sec (float): Seconds since the 1970 epoch.

    Returns:
        datetime: A timezone-aware UTC datetime object.
    """
    return datetime.fromtimestamp(time_sec, tz=timezone.utc)

def get_doy(time_sec: float) -> str:
    """
    Returns the Day of Year (DOY) as a three-digit padded string.

    Args:
        time_sec (float): Unix seconds.

    Returns:
        str: Format 'DDD' (e.g., '001', '365').
    """
    dt = get_datetime(time_sec)
    doy = dt.timetuple().tm_yday
    return f"{doy:03d}"

def get_current_datetime() -> datetime:
    """Returns the current system time as a UTC datetime object."""
    return datetime.now(timezone.utc)

def get_current_day_of_week() -> int:
    """
    Returns the current day of the week as an integer (Sunday=1, Monday=2, etc.).

    Returns:
        int: Day index following Java Calendar convention.
    """
    dt = get_current_datetime()
    py_weekday = dt.weekday()
    return (py_weekday + 2) % 7 if (py_weekday + 2) % 7 != 0 else 7

def get_time_tag_from_seconds(s: float) -> str:
    """
    Converts Unix seconds to a full AIMS time tag.

    Args:
        s (float): Unix seconds.

    Returns:
        str: Format 'YYYY/DDD/HH:MM:SS.mmm'.
    """
    dt = get_datetime(s)
    return get_string_format(dt)

def get_simple_time_tag_from_seconds(s: float) -> str:
    """
    Converts Unix seconds to a simple numeric time tag.

    Args:
        s (float): Unix seconds.

    Returns:
        str: Format 'YYYYDDDHHMMSS'.
    """
    dt = get_datetime(s)
    return get_simple_string_format(dt)

def get_simple_time_tag_from_milliseconds(ms: int) -> str:
    """
    Converts Unix milliseconds to a simple numeric time tag.

    Args:
        ms (int): Unix milliseconds.

    Returns:
        str: Format 'YYYYDDDHHMMSS'.
    """
    dt = get_datetime(ms / 1000.0)
    return get_simple_string_format(dt)

def get_date_format_from_seconds(s: float) -> str:
    """
    Converts Unix seconds to an ISO 8601 formatted string with millisecond precision.

    Args:
        s (float): Unix seconds.

    Returns:
        str: Format 'YYYY-MM-DDTHH:MM:SS.sssZ'.
    """
    dt = get_datetime(s)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

def get_string_format(dt: datetime) -> str:
    """
    Formats a datetime object into a standard AIMS string.

    Args:
        dt (datetime): The source datetime.

    Returns:
        str: Format 'YYYY/DDD/HH:MM:SS.mmm'.
    """
    year = dt.year
    doy = dt.timetuple().tm_yday
    hour = dt.hour
    minute = dt.minute
    second = dt.second
    millisecond = int(dt.microsecond / 1000)
    return f"{year}/{doy:03d}/{hour:02d}:{minute:02d}:{second:02d}.{millisecond:03d}"

def get_simple_string_format(dt: datetime) -> str:
    """
    Formats a datetime object into a simple numeric string.

    Args:
        dt (datetime): The source datetime.

    Returns:
        str: Format 'YYYYDDDHHMMSS'.
    """
    year = dt.year
    doy = dt.timetuple().tm_yday
    hour = dt.hour
    minute = dt.minute
    second = dt.second
    return f"{year}{doy:03d}{hour:02d}{minute:02d}{second:02d}"

def get_year_day_string(s: float) -> str:
    """Returns 'YYYY/DDD' for a given timestamp."""
    dt = get_datetime(s)
    return f"{dt.year}/{dt.timetuple().tm_yday:03d}"

def get_simple_year_day_string(s: float) -> str:
    """Returns 'YYYYDDD' for a given timestamp."""
    dt = get_datetime(s)
    return f"{dt.year}{dt.timetuple().tm_yday:03d}"

def get_time_from_simple_time_string(t: str) -> float:
    """
    Converts 'YYYYDDDHHMMSS' string to seconds since the J2000 epoch.

    Args:
        t (str): Simple time tag.

    Returns:
        float: Seconds since J2000 (2000-01-01T12:00:00Z).
    """
    year = int(t[0:4])
    doy = int(t[4:7])
    hour = int(t[7:9])
    minute = int(t[9:11])
    second = int(t[11:13])
    
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
    dt = dt.replace(hour=hour, minute=minute, second=second)
    
    return dt.timestamp() - TIMEJ2000

def get_day_start(current: float) -> float:
    """
    Retrieves the timestamp for the beginning of the day (00:00:00) 
    for the provided time.

    Args:
        current (float): Absolute timestamp.

    Returns:
        float: Timestamp of day start.
    """
    dt = get_datetime(current)
    start_of_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_of_day.timestamp()

def get_current_day_start() -> float:
    """
    Calculates the beginning of the current system day in seconds 
    relative to the J2000 epoch.

    Returns:
        float: Seconds since J2000 for today at 00:00:00.
    """
    current_time = datetime.now(timezone.utc).timestamp() - TIMEJ2000
    return get_day_start(current_time)

def get_day_end(current: float) -> float:
    """
    Retrieves the timestamp for the start of the next day.

    Args:
        current (float): Absolute timestamp.

    Returns:
        float: Timestamp of day end.
    """
    num_days = math.ceil(current / DAY_IN_SECONDS)
    return num_days * DAY_IN_SECONDS

def get_day_of_year(time_sec: float) -> int:
    """Returns the integer Day of Year (1-366)."""
    return get_datetime(time_sec).timetuple().tm_yday

def get_year(time_sec: float) -> int:
    """Returns the integer year."""
    return get_datetime(time_sec).year

def get_time_tag_from_milliseconds_vax(ms: int, vax_time: bool) -> str:
    """
    Converts Unix milliseconds to AIMS string, with optional 20-year 
    Vax-time correction.

    Args:
        ms (int): Unix milliseconds.
        vax_time (bool): If True, subtracts 20 years from the result.

    Returns:
        str: Format 'YYYY/DDD/HH:MM:SS.mmm'.
    """
    dt = get_datetime(ms / 1000.0)
    if vax_time:
        try:
            dt = dt.replace(year=dt.year - 20)
        except ValueError:
            dt = dt.replace(year=dt.year - 20, day=28)
            
    return get_string_format(dt)

def get_day_of_the_week() -> str:
    """Returns the name of the current day of the week (e.g., 'Monday')."""
    dt = datetime.now(timezone.utc)
    idx = (dt.weekday() + 1) % 7
    return WEEK[idx]

def get_offset(time_string: str) -> float:
    """
    Calculates the duration in seconds between the provided AIMS time 
    string and the current system time.

    Args:
        time_string (str): The AIMS time tag.

    Returns:
        float: Offset in seconds.
    """
    _time = get_time_from_string(time_string)
    current_time = datetime.now(timezone.utc).timestamp()
    return current_time - _time

def is_full_length(time_string: str) -> bool:
    """
    Checks if an AIMS time string contains all necessary date/time components.

    Args:
        time_string (str): The string to check.

    Returns:
        bool: True if it matches recognized complete pattern lengths.
    """
    if is_time_pattern(time_string):
        length = len(time_string)
        return length in [4, 8, 10, 13, 16, 21]
    return False

def get_hours(input_ms: int) -> int:
    """
    Truncates a millisecond timestamp to the start of its hour.

    Args:
        input_ms (int): Unix milliseconds.

    Returns:
        int: Milliseconds at the hour boundary.
    """
    dt = get_datetime(input_ms / 1000.0)
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return int(dt.timestamp() * 1000)

def get_yesterday() -> int:
    """Returns the integer Day of Year for yesterday."""
    dt = datetime.now(timezone.utc) - timedelta(seconds=DAY_IN_SECONDS)
    return dt.timetuple().tm_yday

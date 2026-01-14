"""
Time Utilities
Helper functions for time operations, timestamps, and durations
"""
from datetime import datetime, timedelta, timezone
from typing import Optional
import time


def now() -> datetime:
    """
    Get current UTC time

    Returns:
        Current datetime in UTC
    """
    return datetime.now(timezone.utc)


def utc_now() -> datetime:
    """
    Get current UTC time (alias for clarity)

    Returns:
        Current datetime in UTC
    """
    return datetime.utcnow()


def timestamp() -> float:
    """
    Get current Unix timestamp

    Returns:
        Unix timestamp (seconds since epoch)
    """
    return time.time()


def timestamp_ms() -> int:
    """
    Get current Unix timestamp in milliseconds

    Returns:
        Unix timestamp in milliseconds
    """
    return int(time.time() * 1000)


def iso_now() -> str:
    """
    Get current time as ISO 8601 string

    Returns:
        ISO 8601 formatted timestamp

    Examples:
        >>> iso_now()
        "2026-01-09T15:30:45.123456"
    """
    return utc_now().isoformat()


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime as string

    Args:
        dt: Datetime to format
        format_str: Format string (default: "YYYY-MM-DD HH:MM:SS")

    Returns:
        Formatted datetime string
    """
    return dt.strftime(format_str)


def parse_iso(iso_string: str) -> datetime:
    """
    Parse ISO 8601 datetime string

    Args:
        iso_string: ISO 8601 formatted string

    Returns:
        Parsed datetime

    Examples:
        >>> parse_iso("2026-01-09T15:30:45.123456")
        datetime.datetime(2026, 1, 9, 15, 30, 45, 123456)
    """
    return datetime.fromisoformat(iso_string)


def parse_timestamp(ts: float) -> datetime:
    """
    Convert Unix timestamp to datetime

    Args:
        ts: Unix timestamp (seconds)

    Returns:
        Datetime in UTC
    """
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def to_timestamp(dt: datetime) -> float:
    """
    Convert datetime to Unix timestamp

    Args:
        dt: Datetime to convert

    Returns:
        Unix timestamp (seconds)
    """
    return dt.timestamp()


def duration_seconds(start: datetime, end: Optional[datetime] = None) -> float:
    """
    Calculate duration in seconds

    Args:
        start: Start datetime
        end: End datetime (default: now)

    Returns:
        Duration in seconds
    """
    if end is None:
        end = utc_now()

    delta = end - start
    return delta.total_seconds()


def duration_ms(start: datetime, end: Optional[datetime] = None) -> float:
    """
    Calculate duration in milliseconds

    Args:
        start: Start datetime
        end: End datetime (default: now)

    Returns:
        Duration in milliseconds
    """
    return duration_seconds(start, end) * 1000


def format_duration(seconds: float) -> str:
    """
    Format duration as human-readable string

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string

    Examples:
        >>> format_duration(125.5)
        "2m 5.5s"
        >>> format_duration(3665)
        "1h 1m 5s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds:.1f}s"
        return f"{minutes}m"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60

    if remaining_minutes > 0:
        return f"{hours}h {remaining_minutes}m"
    return f"{hours}h"


def add_seconds(dt: datetime, seconds: float) -> datetime:
    """
    Add seconds to datetime

    Args:
        dt: Base datetime
        seconds: Seconds to add

    Returns:
        New datetime
    """
    return dt + timedelta(seconds=seconds)


def add_minutes(dt: datetime, minutes: float) -> datetime:
    """
    Add minutes to datetime

    Args:
        dt: Base datetime
        minutes: Minutes to add

    Returns:
        New datetime
    """
    return dt + timedelta(minutes=minutes)


def add_hours(dt: datetime, hours: float) -> datetime:
    """
    Add hours to datetime

    Args:
        dt: Base datetime
        hours: Hours to add

    Returns:
        New datetime
    """
    return dt + timedelta(hours=hours)


def add_days(dt: datetime, days: int) -> datetime:
    """
    Add days to datetime

    Args:
        dt: Base datetime
        days: Days to add

    Returns:
        New datetime
    """
    return dt + timedelta(days=days)


def is_expired(dt: datetime, expiry_seconds: float) -> bool:
    """
    Check if datetime has expired

    Args:
        dt: Datetime to check
        expiry_seconds: Expiry duration in seconds

    Returns:
        True if expired
    """
    expiry_time = add_seconds(dt, expiry_seconds)
    return utc_now() > expiry_time


def time_until(target: datetime) -> float:
    """
    Calculate seconds until target datetime

    Args:
        target: Target datetime

    Returns:
        Seconds until target (negative if in past)
    """
    delta = target - utc_now()
    return delta.total_seconds()


def time_since(dt: datetime) -> float:
    """
    Calculate seconds since datetime

    Args:
        dt: Datetime to measure from

    Returns:
        Seconds since datetime
    """
    return duration_seconds(dt)


def sleep(seconds: float) -> None:
    """
    Sleep for specified seconds

    Args:
        seconds: Duration to sleep
    """
    time.sleep(seconds)


def get_date_string(dt: Optional[datetime] = None) -> str:
    """
    Get date as string (YYYY-MM-DD)

    Args:
        dt: Datetime (default: now)

    Returns:
        Date string

    Examples:
        >>> get_date_string()
        "2026-01-09"
    """
    if dt is None:
        dt = utc_now()

    return dt.strftime("%Y-%m-%d")


def get_time_string(dt: Optional[datetime] = None) -> str:
    """
    Get time as string (HH:MM:SS)

    Args:
        dt: Datetime (default: now)

    Returns:
        Time string

    Examples:
        >>> get_time_string()
        "15:30:45"
    """
    if dt is None:
        dt = utc_now()

    return dt.strftime("%H:%M:%S")


def get_datetime_string(dt: Optional[datetime] = None) -> str:
    """
    Get datetime as string (YYYY-MM-DD HH:MM:SS)

    Args:
        dt: Datetime (default: now)

    Returns:
        Datetime string

    Examples:
        >>> get_datetime_string()
        "2026-01-09 15:30:45"
    """
    if dt is None:
        dt = utc_now()

    return dt.strftime("%Y-%m-%d %H:%M:%S")


class Timer:
    """
    Simple timer for measuring elapsed time

    Usage:
        timer = Timer()
        timer.start()
        # ... do work ...
        elapsed = timer.elapsed()
        print(f"Took {elapsed:.2f} seconds")
    """

    def __init__(self):
        """Initialize timer"""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        """Start timer"""
        self.start_time = time.time()
        self.end_time = None

    def stop(self) -> float:
        """
        Stop timer

        Returns:
            Elapsed seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        self.end_time = time.time()
        return self.elapsed()

    def elapsed(self) -> float:
        """
        Get elapsed time

        Returns:
            Elapsed seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    def reset(self):
        """Reset timer"""
        self.start_time = None
        self.end_time = None

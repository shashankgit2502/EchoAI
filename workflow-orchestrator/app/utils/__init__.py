"""
Utilities Module
Shared utility functions for the application
"""
from app.utils.ids import (
    generate_uuid,
    generate_workflow_id,
    generate_agent_id,
    generate_execution_id,
    generate_run_id,
    generate_checkpoint_id,
    generate_tool_call_id,
    extract_workflow_id_from_execution,
    is_valid_uuid,
    generate_id
)

from app.utils.json_utils import (
    load_json,
    save_json,
    save_json_atomic,
    validate_json_schema,
    merge_json,
    json_to_string,
    string_to_json,
    add_metadata,
    strip_metadata,
    format_json_for_display
)

from app.utils.time import (
    now,
    utc_now,
    timestamp,
    timestamp_ms,
    iso_now,
    format_datetime,
    parse_iso,
    parse_timestamp,
    to_timestamp,
    duration_seconds,
    duration_ms,
    format_duration,
    add_seconds,
    add_minutes,
    add_hours,
    add_days,
    is_expired,
    time_until,
    time_since,
    sleep,
    get_date_string,
    get_time_string,
    get_datetime_string,
    Timer
)

__all__ = [
    # IDs
    "generate_uuid",
    "generate_workflow_id",
    "generate_agent_id",
    "generate_execution_id",
    "generate_run_id",
    "generate_checkpoint_id",
    "generate_tool_call_id",
    "extract_workflow_id_from_execution",
    "is_valid_uuid",
    "generate_id",

    # JSON
    "load_json",
    "save_json",
    "save_json_atomic",
    "validate_json_schema",
    "merge_json",
    "json_to_string",
    "string_to_json",
    "add_metadata",
    "strip_metadata",
    "format_json_for_display",

    # Time
    "now",
    "utc_now",
    "timestamp",
    "timestamp_ms",
    "iso_now",
    "format_datetime",
    "parse_iso",
    "parse_timestamp",
    "to_timestamp",
    "duration_seconds",
    "duration_ms",
    "format_duration",
    "add_seconds",
    "add_minutes",
    "add_hours",
    "add_days",
    "is_expired",
    "time_until",
    "time_since",
    "sleep",
    "get_date_string",
    "get_time_string",
    "get_datetime_string",
    "Timer",
]

"""
JSON Utilities
Helper functions for loading, saving, and validating JSON files
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from app.core.logging import get_logger

logger = get_logger(__name__)


def load_json(file_path: Path, default: Optional[Any] = None) -> Any:
    """
    Load JSON from file with error handling

    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist or invalid

    Returns:
        Parsed JSON data or default

    Raises:
        FileNotFoundError: If file not found and no default provided
        json.JSONDecodeError: If JSON is invalid and no default provided
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON from: {file_path}")
        return data

    except FileNotFoundError:
        if default is not None:
            logger.warning(f"File not found: {file_path}, returning default")
            return default
        logger.error(f"File not found: {file_path}")
        raise

    except json.JSONDecodeError as e:
        if default is not None:
            logger.warning(f"Invalid JSON in {file_path}: {e}, returning default")
            return default
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise


def save_json(
    file_path: Path,
    data: Any,
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    Save data as JSON file with pretty printing

    Args:
        file_path: Path to save JSON file
        data: Data to save
        indent: Indentation spaces (default: 2)
        ensure_ascii: Escape non-ASCII characters (default: False)

    Raises:
        IOError: If unable to write file
    """
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

        logger.debug(f"Saved JSON to: {file_path}")

    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}", exc_info=True)
        raise


def save_json_atomic(
    file_path: Path,
    data: Any,
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    Save JSON atomically (temp file + rename)

    Ensures file integrity by writing to temp file first,
    then renaming to target path

    Args:
        file_path: Path to save JSON file
        data: Data to save
        indent: Indentation spaces
        ensure_ascii: Escape non-ASCII characters
    """
    temp_path = file_path.with_suffix('.tmp')

    try:
        # Write to temp file
        save_json(temp_path, data, indent, ensure_ascii)

        # Atomic rename
        temp_path.replace(file_path)

        logger.debug(f"Atomically saved JSON to: {file_path}")

    except Exception as e:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()

        logger.error(f"Failed to atomically save JSON to {file_path}: {e}", exc_info=True)
        raise


def validate_json_schema(data: Dict, required_keys: list) -> bool:
    """
    Validate JSON has required keys

    Args:
        data: JSON data
        required_keys: List of required key names

    Returns:
        True if valid

    Raises:
        ValueError: If required keys missing
    """
    missing_keys = [key for key in required_keys if key not in data]

    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    return True


def merge_json(base: Dict, update: Dict, deep: bool = True) -> Dict:
    """
    Merge two JSON objects

    Args:
        base: Base JSON object
        update: Updates to apply
        deep: Deep merge (default: True)

    Returns:
        Merged JSON object
    """
    if not deep:
        return {**base, **update}

    # Deep merge
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_json(result[key], value, deep=True)
        else:
            result[key] = value

    return result


def json_to_string(data: Any, pretty: bool = True, indent: int = 2) -> str:
    """
    Convert data to JSON string

    Args:
        data: Data to convert
        pretty: Pretty print (default: True)
        indent: Indentation spaces

    Returns:
        JSON string
    """
    if pretty:
        return json.dumps(data, indent=indent, ensure_ascii=False)
    else:
        return json.dumps(data, ensure_ascii=False)


def string_to_json(json_string: str) -> Any:
    """
    Parse JSON string

    Args:
        json_string: JSON string

    Returns:
        Parsed data

    Raises:
        json.JSONDecodeError: If invalid JSON
    """
    return json.loads(json_string)


def add_metadata(data: Dict) -> Dict:
    """
    Add metadata to JSON object

    Args:
        data: JSON object

    Returns:
        JSON object with metadata
    """
    if "metadata" not in data:
        data["metadata"] = {}

    data["metadata"]["updated_at"] = datetime.utcnow().isoformat()

    return data


def strip_metadata(data: Dict) -> Dict:
    """
    Remove metadata from JSON object

    Args:
        data: JSON object

    Returns:
        JSON object without metadata
    """
    result = data.copy()

    if "metadata" in result:
        del result["metadata"]

    return result


def format_json_for_display(data: Any, max_depth: int = 3) -> str:
    """
    Format JSON for display with depth limit

    Args:
        data: JSON data
        max_depth: Maximum nesting depth to display

    Returns:
        Formatted JSON string
    """
    def _truncate(obj, depth):
        if depth > max_depth:
            if isinstance(obj, dict):
                return "{...}"
            elif isinstance(obj, list):
                return "[...]"
            else:
                return str(obj)

        if isinstance(obj, dict):
            return {k: _truncate(v, depth + 1) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_truncate(item, depth + 1) for item in obj]
        else:
            return obj

    truncated = _truncate(data, 0)
    return json.dumps(truncated, indent=2, ensure_ascii=False)

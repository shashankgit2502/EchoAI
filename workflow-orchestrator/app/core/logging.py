"""
Logging Configuration
"""
import logging
import sys
from pathlib import Path
from app.core.config import get_settings


def setup_logging():
    """Configure application-wide logging"""
    settings = get_settings()

    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format=settings.LOG_FORMAT,
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),
            # File handler
            logging.FileHandler(log_dir / "workflow_orchestrator.log")
        ]
    )

    # Set level for external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get logger for a specific module"""
    return logging.getLogger(name)

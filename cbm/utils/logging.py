"""Logging configuration for charting-by-machines."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Configure structured logging with loguru.
    
    Parameters
    ----------
    level : str
        Log level ("DEBUG", "INFO", "WARNING", "ERROR").
    log_file : str, optional
        Path to log file. If None, logs only to console.
    rotation : str
        Log file rotation size.
    retention : str
        How long to keep old log files.
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with colors
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=rotation,
            retention=retention,
            compression="zip",
        )
    
    logger.debug(f"Logging configured: level={level}, file={log_file}")


def get_logger(name: str):
    """
    Get a logger instance with the given name.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__).
        
    Returns
    -------
    loguru.Logger
        Logger instance.
    """
    return logger.bind(name=name)

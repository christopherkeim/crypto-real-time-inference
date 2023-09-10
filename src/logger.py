"""
Creates a console logger with formatting.
"""

from typing import Optional
import logging


def get_console_logger(name: Optional[str] = "base") -> logging.Logger:
    # Create a logger if it doesn't exist
    logger: logging.RootLogger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create the console handler with formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(console_handler)

    return logger

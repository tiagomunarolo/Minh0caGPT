import logging
from logging import Logger
from typing import Optional


class CustomLogger:
    def __init__(self, name: str = __name__, log_file: Optional[str] = None, level: int = logging.DEBUG):
        """
        Initialize the custom logger with optional file logging and specified logging level.

        :param name: Name of the logger (usually the module name)
        :param log_file: File path to log to (optional). If not provided, logs will only appear in the console.
        :param level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add console handler to logger
        self.logger.addHandler(console_handler)

        if log_file:
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)

            # Add file handler to logger
            self.logger.addHandler(file_handler)

    def get_logger(self) -> Logger:
        """
        Returns the logger instance to log messages.

        :return: Configured logger
        """
        return self.logger

import logging
import os


def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger with the specified name, log file, and log level.

    Parameters:
    - name (str): Name of the logger.
    - log_file (str): Path to the log file.
    - level (int): Logging level.

    Returns:
    - logging.Logger: Configured logger instance.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    # Set log format with seconds in the timestamp
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Create logger and set level
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
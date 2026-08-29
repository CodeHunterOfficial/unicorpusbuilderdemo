# logger_setup.py
import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Create a logger that writes to console AND optionally to a file.
    Useful for interactive debugging.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    if level == "OFF":
        logger.addHandler(logging.NullHandler())
        return logger

    logger.setLevel(getattr(logging, level, logging.INFO))

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
        logger.addHandler(file_handler)

    return logger


def get_file_logger(name: str, log_file: str) -> logging.Logger:
    """
    Create a logger that writes ONLY to a file, no console output.
    Suitable for modules where console output should remain clean.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    if level == "OFF":
        logger.addHandler(logging.NullHandler())
        return logger

    logger.setLevel(getattr(logging, level, logging.INFO))

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(file_handler)

    return logger
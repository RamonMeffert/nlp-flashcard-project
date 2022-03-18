import logging
import os

from dotenv import load_dotenv

load_dotenv()


def get_logger():
    # creates a default logger for the project
    logger = logging.getLogger("Flashcards")

    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger.setLevel(log_level)

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # file handler
    fh = logging.FileHandler("logs.log")
    fh.setFormatter(formatter)

    # stout
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

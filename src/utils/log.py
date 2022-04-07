import coloredlogs
import logging
import os

from dotenv import load_dotenv

load_dotenv()

# creates a default logger for the project. We declare it in the global scope
# so it acts like a singleton
logger = logging.getLogger("Flashcards")

log_level = os.getenv("LOG_LEVEL", "INFO")
logger.setLevel(log_level)

# Log format
formatter = coloredlogs.ColoredFormatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# stout
ch = logging.StreamHandler()
ch.setFormatter(formatter)

# colored output so log messages stand out more
# coloredlogs.install(level=log_level, logger=logger)

# file handler
fh = logging.FileHandler("logs.log")
fh.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

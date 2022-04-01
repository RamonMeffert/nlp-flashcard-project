import time
from typing import Dict
from dotenv import load_dotenv
import os
from src.utils.log import get_logger


logger = get_logger()


load_dotenv()


ENABLE_TIMING = os.getenv("ENABLE_TIMING", "false").lower() == "true"

if ENABLE_TIMING:
    logger.info("Timing is enabled")


TimingType = Dict[str, float]

TIMES: TimingType = {}


def timeit(name: str):
    def _timeit(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            if name not in TIMES:
                TIMES[name] = []
            TIMES[name].append(end - start)

            return result

        if ENABLE_TIMING:
            return wrapper
        return func
    return _timeit


def get_times() -> TimingType:
    _warn_if_timing_disabled()
    return TIMES


def reset_times() -> None:
    _warn_if_timing_disabled()
    TIMES.clear()


def _warn_if_timing_disabled() -> None:
    if not ENABLE_TIMING:
        logger.warning(
            "Timing is disabled, please set ENABLE_TIMING to true in the .env file")

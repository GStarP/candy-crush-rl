import time
import logging
from functools import wraps


def log_exec_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        cost = end_time - start_time
        logging.info(f"log_exec_time： func={func.__name__}, cost={cost:.4f}s")
        return result

    return wrapper

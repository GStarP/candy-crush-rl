import time
import logging
from functools import wraps
import os


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


def configure_logging(work_dir: str, debug=False):
    logging.basicConfig(
        filename=os.path.join(work_dir, "env.log"),
        encoding="utf-8",
        level=logging.INFO if not debug else logging.DEBUG,
        format="%(asctime)s.%(msecs)03d %(levelname)s [pid=%(process)d] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

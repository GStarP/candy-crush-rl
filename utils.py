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
        if cost >= 0.01:
            logging.warning(f"log_exec_time： func={func.__name__}, cost={cost:.4f}s")
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


def windows_notify(title: str, msg: str):
    from win10toast_persist import ToastNotifier

    try:
        toaster = ToastNotifier()
        toaster.show_toast(title, msg, duration=10, threaded=True)
    except Exception as e:
        logging.exception(f"windows_notify_error: {e}")

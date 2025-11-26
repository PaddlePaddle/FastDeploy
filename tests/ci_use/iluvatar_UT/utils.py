import functools
import signal


def timeout(seconds):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds} seconds")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
                signal.alarm(0)
                return result
            finally:
                signal.signal(signal.SIGALRM, original_handler)
                signal.alarm(0)

        return wrapper

    return decorator


TIMEOUT_MSG = "The timeout exit may be due to multiple processes sharing the same gpu card. You can check this using ixsmi on the device."

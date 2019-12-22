import logging

from functools import wraps


def set_log_handler(func):
    @wraps(func)
    def with_set_log_handler(*args, **kwargs):
        log_setup = kwargs.pop('log_setup', None)
        if log_setup is None:
            return func(*args, **kwargs)

        handl = log_setup()
        res = func(*args, **kwargs)

        # remove unused handler
        logger = logging.getLogger("genens")
        logger.removeHandler(handl)
        return res

    return with_set_log_handler

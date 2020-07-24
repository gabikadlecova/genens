import json
import logging
import logging.config

from functools import wraps
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager


def set_log_handler(func):
    @wraps(func)
    def with_set_log_handler(*args, **kwargs):
        log_setup = kwargs.pop('log_setup', None)
        if log_setup is not None:
            log_setup()

        return func(*args, **kwargs)

    return with_set_log_handler


class GenensLogger:
    def __init__(self, config_path, n_jobs=1):
        self.n_jobs = n_jobs

        with open(config_path, 'r') as f:
            config = json.load(f)

        self._log_queue = None
        self._logging_config = config
        logging.config.dictConfig(config)

        logger = logging.getLogger("genens")
        for handler in logger.handlers:
            print(handler.level)

        # special setup for multiprocessing
        if self.n_jobs != 1:
            mp_manager = Manager()
            self._log_queue = mp_manager.Queue()

            if "handlers" in config:
                config.pop("handlers")
                self._logging_config = config

            config["loggers"]["genens"].pop("handlers")

            handler = QueueHandler(self._log_queue)
            logger = logging.getLogger("genens")
            logger.addHandler(handler)

    def listen(self):
        if self.n_jobs == 1:
            return None

        logger = logging.getLogger("genens")

        # handlers receive messages from queue
        handlers = []
        for handler in logger.handlers:
            if not isinstance(handler, QueueHandler):
                handlers.append(handler)

        for handler in handlers:
            logger.handlers.remove(handler)

        queue_listener = QueueListener(self._log_queue, *handlers, respect_handler_level=True)
        queue_listener.start()
        return queue_listener, handlers

    def close(self, log_handlers=None):
        if log_handlers is None:
            return

        queue_listener, queue_handlers = log_handlers
        if queue_listener is None:
            return

        queue_listener.stop()

        # set handlers back
        logger = logging.getLogger("genens")
        for handler in queue_handlers:
            print(handler.level)
            logger.addHandler(handler)

    def setup_child_logging(self):
        if self.n_jobs == 1:
            return

        logger = logging.getLogger("genens")

        if not logger.hasHandlers():
            logging.config.dictConfig(self._logging_config)
            handler = QueueHandler(self._log_queue)
            logger.addHandler(handler)

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
            self.config = json.load(f)

        self._log_queue = None
        self._logging_config = self.config
        logging.config.dictConfig(self.config)

        # special setup for multiprocessing
        if self.n_jobs != 1:
            mp_manager = Manager()
            self._log_queue = mp_manager.Queue()

            if "handlers" in self.config:
                self.config.pop("handlers")
                self._logging_config = self.config

            handler = QueueHandler(self._log_queue)
            logger = logging.getLogger("genens")
            logger.addHandler(handler)

    # TODO need the queue so that I dont remove the handlers (does not work well)
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

        queue_listener = QueueListener(self._log_queue, *handlers)
        queue_listener.start()
        return queue_listener

    def close(self, queue_listener: QueueListener = None):
        if queue_listener is None:
            return

        queue_listener.stop()

        handlers = []
        # set handlers back
        logger = logging.getLogger("genens")
        for handler in queue_listener.:
            logger.addHandler(handler)

        for handler in handlers:
            logger.removeHandler(handler)

    def setup_child_logging(self):
        if self.n_jobs == 1:
            return

        logger = logging.getLogger("genens")

        if not logger.hasHandlers():
            logging.config.dictConfig(self._logging_config)
            handler = QueueHandler(self._log_queue)
            logger.addHandler(handler)

# mneme/replay/logging.py
import logging

# Library logger: silent by default
logger = logging.getLogger("mneme.replay")
logger.addHandler(logging.NullHandler())  # avoids "No handler" warnings
logger.propagate = False  # don't bubble up to root logger

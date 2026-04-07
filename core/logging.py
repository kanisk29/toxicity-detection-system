import logging
import sys

logger = logging.getLogger("toxicity-backend")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

logger.propagate = False
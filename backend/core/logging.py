import logging

logger = logging.getLogger("ToxScan")
logger.setLevel(logging.DEBUG)
formatter = logger.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler(filename="logs/app.log")
file_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
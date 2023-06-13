import sys

import logging


COLOR_SEQ = "\033[38;5;%dm"
RESET_SEQ = "\033[0;0m"

LVLNAME = {
    'DEBUG': 8,
    'INFO': 2,
    'WARNING': 3,
    'ERROR': 1,
    'CRITICAL': 9,
}

COLORS = {
    'green': 2,
    'yellow': 3,
    'red': 1,
    'obs': 249
}

FORMAT = '[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s'


def logger(name, level='INFO'):
    log = logging.getLogger(name)
    formatter = ColoredFormatter()

    log.setLevel(logging.DEBUG)
    handler_stream = logging.StreamHandler(stream=sys.stdout)
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(level)
    log.addHandler(handler_stream)

    return log, formatter


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg=FORMAT):
        logging.Formatter.__init__(self, msg)

    def format(self, record):
        levelname = record.levelname
        levelname_color = COLOR_SEQ % (LVLNAME[levelname]) + levelname + RESET_SEQ
        record.levelname = levelname_color

        return logging.Formatter.format(self, record)

    @staticmethod
    def colored_logs(text, color):
        return COLOR_SEQ % COLORS[color] + str(text) + RESET_SEQ

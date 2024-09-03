import logging
import colorlog

logger = logging.getLogger('color_log')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

_FORMAT = "%(log_color)s%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

formatter = colorlog.ColoredFormatter(
    _FORMAT,
    datefmt=_DATE_FORMAT,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)
handler.setFormatter(formatter)

logger.addHandler(handler)

def init_clogger():
    return logger
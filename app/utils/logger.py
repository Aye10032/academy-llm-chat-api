import logging
import sys
from types import FrameType
from typing import cast

from loguru import logger
from datetime import datetime

from app.core.config import get_settings


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage(),
        )


now = datetime.now()
year = now.year
month = now.month
day = now.day

loguru_config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "level": get_settings().LOGGING_LEVEL,
            "format": "<green>{time:HH:mm}</green> | <level>{level}</level> | "
                      "<cyan>{module}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        },
        {
            "sink": f'log/runtime_{year}{month:02d}{day:02d}.log',
            "level": get_settings().LOGGING_LEVEL,
            "rotation": "10 MB",
            "retention": "1 week",
            "encoding": 'utf-8',
            "format": "{time:YYYY-mm-dd HH:mm:ss.SSS} | {thread.name} | {level} | {module} : {function}:{line} -  {message}"
        },
        {
            "sink": f'log/error/error_{year}{month:02d}{day:02d}.log',
            "level": 'ERROR',
            "retention": "1 week",
            "rotation": "10 MB",
            "encoding": 'utf-8',
            "format": "{time:YYYY-mm-dd HH:mm:ss.SSS} | {thread.name} | {level} | {module} : {function}:{line} -  {message}"
        },
    ],
}


def init_logging():
    logger_names = ["uvicorn.asgi", "uvicorn.access", "uvicorn"]

    # change handler for default uvicorn logger
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in logger_names:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]

    logger.configure(**loguru_config)

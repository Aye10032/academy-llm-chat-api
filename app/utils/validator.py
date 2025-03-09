import re
from functools import wraps
from typing import Callable, TypeVar

from loguru import logger

T = TypeVar('T')


class InputValidationError(Exception):
    pass


def simple_char_valid(text: str) -> bool:
    return bool(re.match(r'^[A-Za-z0-9_]+$', text))


def validate_input(validator: Callable[[str], bool], error_msg: str):
    """检查输入参数是否符合要求

    Args:
        validator: 用于进行检测的函数，返回一个布尔值
        error_msg: 错误信息

    Returns:
        被装饰的函数

    Raises:
        InputValidationError: 当输入格式错误时，抛出此错误
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            while True:
                try:
                    result = func(*args, **kwargs)
                    if not validator(result):
                        raise InputValidationError(error_msg)
                    return result
                except InputValidationError as e:
                    logger.warning(str(e))

        return wrapper

    return decorator

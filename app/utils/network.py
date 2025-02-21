import random
import re
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import requests
from loguru import logger
from pydantic import AnyHttpUrl
from requests import HTTPError, ReadTimeout
from requests.exceptions import ChunkedEncodingError
from urllib3.exceptions import ResponseError


def retry(retries: int = 3, delay: float = 1) -> Callable:
    """为函数提供重试逻辑的装饰器

    Args:
        retries: 最大重试次数，默认为3
        delay: 两次重试之间的延迟时间（秒），默认为1

    Returns:
        Callable: 被装饰的函数

    Raises:
        ValueError: 如果retries小于1或delay小于等于0，则抛出此异常
    """
    if retries < 1 or delay <= 0:
        raise ValueError('Wrong param')

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for i in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except (
                    ResponseError,
                    ReadTimeout,
                    ChunkedEncodingError,
                    ConnectionError,
                    ConnectionError,
                ) as e:
                    if i == retries:
                        logger.error(f'Error: {repr(e)}.')
                        logger.error(
                            f'"{func.__name__}()" failed after {retries} retries.'
                        )
                        break
                    else:
                        logger.debug(f'Error: {repr(e)} -> Retrying...')
                        time.sleep(delay)

        return wrapper

    return decorator


@retry(delay=random.randint(3, 5))
def download_html(url: AnyHttpUrl) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers, timeout=60)
    if response.status_code != 200:
        raise HTTPError('请求失败')

    return response.text


# Patterns
SCRIPT_PATTERN = r'<[ ]*script.*?\/[ ]*script[ ]*>'
STYLE_PATTERN = r'<[ ]*style.*?\/[ ]*style[ ]*>'
META_PATTERN = r'<[ ]*meta.*?>'
COMMENT_PATTERN = r'<[ ]*!--.*?--[ ]*>'
LINK_PATTERN = r'<[ ]*link.*?>'
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r'(<svg[^>]*>)(.*?)(<\/svg>)'


def _replace_svg(html: str, new_content: str = 'this is a placeholder') -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f'{match.group(1)}{new_content}{match.group(3)}',
        html,
        flags=re.DOTALL,
    )


def _replace_base64_images(html: str, new_image_src: str = '#') -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)


def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
    html = re.sub(
        SCRIPT_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        STYLE_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        META_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        COMMENT_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        LINK_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    if clean_svg:
        html = _replace_svg(html)
    if clean_base64:
        html = _replace_base64_images(html)
    return html

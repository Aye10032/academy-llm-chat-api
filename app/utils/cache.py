import hashlib
import json
import time
from functools import wraps
from typing import Any, Callable, Optional

from loguru import logger
from pydantic import BaseModel


class CachedItem(BaseModel):
    model: Any
    last_used: float
    ttl: Optional[int]


class ModelCache:
    def __init__(self):
        self._cache: dict[str, CachedItem] = {}

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None

        item = self._cache[key]
        current_time = time.time()

        # 检查是否过期
        if item.ttl is not None and current_time - item.last_used > item.ttl:
            del self._cache[key]
            return None

        # 更新最后使用时间
        item.last_used = current_time
        return item.model

    def set(self, key: str, model: Any, ttl: Optional[int] = None):
        self._cache[key] = CachedItem(model=model, last_used=time.time(), ttl=ttl)

    def clear(self):
        self._cache.clear()
        logger.info('已清除所有模型缓存')

    def clear_expired(self):
        """清除所有过期的缓存"""
        current_time = time.time()
        expired_keys = [
            key
            for key, item in self._cache.items()
            if item.ttl is not None and current_time - item.last_used > item.ttl
        ]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.info(f'已清除 {len(expired_keys)} 个过期的模型缓存')


# 全局缓存实例
_model_cache = ModelCache()


def cache_model(ttl: Optional[int] = 3600):
    """模型缓存装饰器

    Args:
        ttl: 过期时间（秒），None表示永不过期，默认一小时
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 创建缓存键
            cache_key_dict = {
                'func_name': func.__name__,
                'args': args,
                'kwargs': {k: str(v) for k, v in kwargs.items()},
            }
            cache_key = hashlib.md5(json.dumps(cache_key_dict, sort_keys=True).encode()).hexdigest()

            # 尝试从缓存获取
            cached_model = _model_cache.get(cache_key)
            if cached_model is not None:
                logger.debug(f'使用缓存的模型实例: {func.__name__}')
                return cached_model

            # 如果缓存中没有，执行原函数并缓存结果
            result = func(*args, **kwargs)
            _model_cache.set(cache_key, result, ttl)
            return result

        return wrapper

    return decorator

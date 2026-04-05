"""Thread-safe Singleton metaclass used by the project.

This helper is deliberately implemented as a top-level module so it can be
imported for lightweight unit tests without importing the full
`WeatherRoutingTool` package (which triggers heavy imports at package init).

The metaclass guarantees that only one instance per class is created even
when multiple threads try to instantiate concurrently.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Type


class SingletonBase(type):
    """Thread-safe Singleton metaclass.

    Usage:
        class MyClass(metaclass=SingletonBase):
            pass

        a = MyClass()
        b = MyClass()
        assert a is b
    """

    _instances: Dict[Type[Any], Any] = {}
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        # Fast path: return existing instance without locking
        if cls in SingletonBase._instances:
            return SingletonBase._instances[cls]

        # Slow path: acquire lock and create instance if still missing
        with SingletonBase._lock:
            if cls not in SingletonBase._instances:
                SingletonBase._instances[cls] = super().__call__(*args, **kwargs)

            return SingletonBase._instances[cls]

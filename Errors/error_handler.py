from functools import wraps
from .logger import logger
from .exceptions import SIRBaseError

def safe_call(default_return=None):
    """
    Decorator universal pentru try/except.
    Daca o functie arunca eroare -> log + return implicit.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except SIRBaseError as e:
                logger.error(f"{func.__name__} | {e}")
            except Exception as e:
                logger.exception(f"{func.__name__} | Unexpectet error: {e}")
            return default_return
        return wrapper
    return decorator

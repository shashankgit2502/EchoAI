"""
Retry and Timeout Helpers
Bounded retries and timeouts for async validation operations
"""
import asyncio
import time
from typing import Callable, TypeVar, Optional, Any
from functools import wraps

from app.core.constants import SystemLimits
from app.core.logging import get_logger
from app.validator.errors import TimeoutException, RetryExhaustedException

logger = get_logger(__name__)

T = TypeVar('T')


# ============================================================================
# RETRY DECORATOR
# ============================================================================

def with_retry(
    max_attempts: int = SystemLimits.MAX_RETRY_ATTEMPTS,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff

    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to catch and retry

    Usage:
        @with_retry(max_attempts=3, delay_seconds=1.0)
        async def my_function():
            # code that might fail
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay_seconds

            for attempt in range(1, max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(f"{func.__name__} succeeded on attempt {attempt}")
                    return result

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        break

                    logger.warning(
                        f"{func.__name__} failed on attempt {attempt}/{max_attempts}: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_multiplier

            # All attempts failed
            raise RetryExhaustedException(
                f"{func.__name__} failed after {max_attempts} attempts. "
                f"Last error: {last_exception}"
            )

        return wrapper
    return decorator


# ============================================================================
# TIMEOUT DECORATOR
# ============================================================================

def with_timeout(timeout_seconds: float = SystemLimits.DEFAULT_EXECUTION_TIMEOUT_SECONDS):
    """
    Timeout decorator for async functions

    Args:
        timeout_seconds: Timeout in seconds

    Usage:
        @with_timeout(timeout_seconds=30)
        async def my_function():
            # code that might take too long
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
                return result

            except asyncio.TimeoutError:
                logger.error(
                    f"{func.__name__} timed out after {timeout_seconds}s"
                )
                raise TimeoutException(
                    f"{func.__name__} exceeded timeout of {timeout_seconds}s"
                )

        return wrapper
    return decorator


# ============================================================================
# COMBINED RETRY + TIMEOUT
# ============================================================================

def with_retry_and_timeout(
    max_attempts: int = SystemLimits.MAX_RETRY_ATTEMPTS,
    timeout_seconds: float = SystemLimits.DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    delay_seconds: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """
    Combined retry and timeout decorator

    Args:
        max_attempts: Maximum retry attempts
        timeout_seconds: Timeout per attempt
        delay_seconds: Initial delay between retries
        exceptions: Exceptions to catch and retry

    Usage:
        @with_retry_and_timeout(max_attempts=3, timeout_seconds=30)
        async def my_function():
            # code that might fail or take too long
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay_seconds

            for attempt in range(1, max_attempts + 1):
                try:
                    # Apply timeout to each attempt
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout_seconds
                    )

                    if attempt > 1:
                        logger.info(f"{func.__name__} succeeded on attempt {attempt}")

                    return result

                except asyncio.TimeoutError:
                    last_exception = TimeoutException(
                        f"{func.__name__} timed out after {timeout_seconds}s"
                    )

                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} timed out on all {max_attempts} attempts"
                        )
                        break

                    logger.warning(
                        f"{func.__name__} timed out on attempt {attempt}/{max_attempts}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        break

                    logger.warning(
                        f"{func.__name__} failed on attempt {attempt}/{max_attempts}: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                await asyncio.sleep(current_delay)
                current_delay *= 2.0

            # All attempts failed
            raise RetryExhaustedException(
                f"{func.__name__} failed after {max_attempts} attempts. "
                f"Last error: {last_exception}"
            )

        return wrapper
    return decorator


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """
    Simple rate limiter for validation operations

    Prevents overwhelming external services (MCP, LLM APIs)
    """

    def __init__(self, calls_per_second: float = 10.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time: Optional[float] = None

    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        if self.last_call_time is not None:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                await asyncio.sleep(wait_time)

        self.last_call_time = time.time()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def run_with_timeout(
    coro,
    timeout_seconds: float,
    operation_name: str = "operation"
) -> Any:
    """
    Run a coroutine with timeout

    Args:
        coro: Coroutine to run
        timeout_seconds: Timeout in seconds
        operation_name: Name for logging

    Returns:
        Result of coroutine

    Raises:
        TimeoutException: If operation times out
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        return result

    except asyncio.TimeoutError:
        logger.error(f"{operation_name} timed out after {timeout_seconds}s")
        raise TimeoutException(
            f"{operation_name} exceeded timeout of {timeout_seconds}s"
        )


async def run_with_retry(
    coro_factory: Callable[[], Any],
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    operation_name: str = "operation"
) -> Any:
    """
    Run a coroutine factory with retry

    Args:
        coro_factory: Function that returns a coroutine
        max_attempts: Maximum retry attempts
        delay_seconds: Initial delay between retries
        operation_name: Name for logging

    Returns:
        Result of coroutine

    Raises:
        RetryExhaustedException: If all retries fail
    """
    last_exception = None
    current_delay = delay_seconds

    for attempt in range(1, max_attempts + 1):
        try:
            result = await coro_factory()
            if attempt > 1:
                logger.info(f"{operation_name} succeeded on attempt {attempt}")
            return result

        except Exception as e:
            last_exception = e

            if attempt == max_attempts:
                logger.error(
                    f"{operation_name} failed after {max_attempts} attempts: {e}"
                )
                break

            logger.warning(
                f"{operation_name} failed on attempt {attempt}/{max_attempts}: {e}. "
                f"Retrying in {current_delay:.1f}s..."
            )

            await asyncio.sleep(current_delay)
            current_delay *= 2.0

    raise RetryExhaustedException(
        f"{operation_name} failed after {max_attempts} attempts. "
        f"Last error: {last_exception}"
    )

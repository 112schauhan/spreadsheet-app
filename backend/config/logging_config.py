"""
Logging configuration for the collaborative spreadsheet backend.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import structlog
from structlog.stdlib import LoggerFactory

from .settings import get_settings

# Global logger cache
_loggers = {}


def setup_logging() -> None:
    """Setup application logging configuration."""
    settings = get_settings()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if not settings.DEBUG
            else structlog.dev.ConsoleRenderer(colors=True)
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))

    if settings.DEBUG:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(settings.LOG_FORMAT)

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if configured)
    if settings.LOG_FILE:
        log_file_path = Path(settings.LOG_FILE)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        if settings.LOG_ROTATION:
            file_handler = logging.handlers.RotatingFileHandler(
                settings.LOG_FILE,
                maxBytes=_parse_size(settings.LOG_MAX_SIZE),
                backupCount=settings.LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(
                settings.LOG_FILE,
                encoding='utf-8'
            )

        file_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
        file_formatter = logging.Formatter(settings.LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Configure specific loggers
    _configure_specific_loggers(settings)

    # Log startup message
    logger = get_logger(__name__)
    logger.info(
        "Logging configured",
        level=settings.LOG_LEVEL,
        debug_mode=settings.DEBUG,
        log_file=settings.LOG_FILE
    )


def _configure_specific_loggers(settings) -> None:
    """Configure specific application loggers."""

    # WebSocket logger
    ws_logger = logging.getLogger("websockets")
    ws_logger.setLevel(
        logging.WARNING if not settings.DEBUG else logging.DEBUG)

    # FastAPI logger
    fastapi_logger = logging.getLogger("fastapi")
    fastapi_logger.setLevel(logging.INFO)

    # Uvicorn loggers
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.setLevel(
        logging.INFO if settings.DEBUG else logging.WARNING)

    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.setLevel(logging.INFO)

    # Application specific loggers
    app_loggers = [
        "handlers",
        "services",
        "models",
        "utils",
        "events",
        "middleware"
    ]

    for logger_name in app_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL))


def _parse_size(size_str: str) -> int:
    """Parse size string (e.g., '10MB') to bytes."""
    size_str = size_str.upper().strip()

    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
    }

    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            number = size_str[:-len(suffix)]
            try:
                return int(float(number) * multiplier)
            except ValueError:
                break

    # Default to bytes if no suffix or invalid format
    try:
        return int(size_str)
    except ValueError:
        return 10 * 1024 * 1024  # Default 10MB


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance."""
    if name not in _loggers:
        _loggers[name] = structlog.get_logger(name)
    return _loggers[name]


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes."""

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def log_function_call(func):
    """Decorator to log function calls with parameters and results."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)

        # Log function entry
        logger.debug(
            f"Calling {func.__name__}",
            function=func.__name__,
            args=str(args)[:200],  # Limit arg string length
            kwargs=str(kwargs)[:200]
        )

        try:
            result = func(*args, **kwargs)
            logger.debug(
                f"Completed {func.__name__}",
                function=func.__name__,
                result_type=type(result).__name__
            )
            return result
        except Exception as e:
            logger.error(
                f"Error in {func.__name__}",
                function=func.__name__,
                error=str(e),
                exc_info=True
            )
            raise

    return wrapper


async def log_async_function_call(func):
    """Decorator to log async function calls with parameters and results."""
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)

        # Log function entry
        logger.debug(
            f"Calling async {func.__name__}",
            function=func.__name__,
            args=str(args)[:200],
            kwargs=str(kwargs)[:200]
        )

        try:
            result = await func(*args, **kwargs)
            logger.debug(
                f"Completed async {func.__name__}",
                function=func.__name__,
                result_type=type(result).__name__
            )
            return result
        except Exception as e:
            logger.error(
                f"Error in async {func.__name__}",
                function=func.__name__,
                error=str(e),
                exc_info=True
            )
            raise

    return wrapper


class PerformanceLogger:
    """Context manager for logging performance metrics."""

    def __init__(self, operation_name: str, logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger(__name__)
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time

        if exc_type is not None:
            self.logger.error(
                f"Failed {self.operation_name}",
                duration=duration,
                error=str(exc_val)
            )
        else:
            self.logger.info(
                f"Completed {self.operation_name}",
                duration=duration
            )


def configure_test_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Reduce noise from external libraries during testing
    logging.getLogger("websockets").setLevel(logging.ERROR)
    logging.getLogger("uvicorn").setLevel(logging.ERROR)

import sys
from pathlib import Path

from loguru import logger

from .config import settings


def configure_logging() -> None:
    """Configure loguru logging based on environment settings."""
    # Remove default handler
    logger.remove()

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Console logging - colorized for development
    if settings.debug:
        logger.add(
            sys.stderr,
            level="DEBUG",
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=True,
        )
    else:
        logger.add(
            sys.stderr,
            level=settings.log_level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
        )

    # File logging with rotation
    logger.add(
        log_dir / "app.log",
        level="INFO",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        rotation="10 MB",
        retention=10,
        compression="zip",
    )

    # Error-specific log file
    logger.add(
        log_dir / "error.log",
        level="ERROR",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message} | "
            "{extra}"
        ),
        rotation="5 MB",
        retention=20,
        compression="zip",
    )

    # JSON logging for production analysis
    if not settings.debug:
        logger.add(
            log_dir / "app.json",
            level="INFO",
            serialize=True,
            rotation="50 MB",
            retention=10,
            compression="zip",
        )

    logger.info("Logging configured successfully")


def get_logger(name: str):
    """Get a logger instance with the given name."""
    return logger.bind(module=name)

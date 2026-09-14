"""structlog configuration.

Two renderers configurable via config.yaml:
  - pretty: rich-rendered ConsoleRenderer for humans during development.
  - json:   JSONRenderer for machine ingestion / future log shipping.

Always logs to a rotating-style file (single file in Phase 0; phase 4 will rotate).
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path

import structlog

from core.config import LoggingConfig


def configure_logging(cfg: LoggingConfig, project_root: Path) -> None:
    log_file = project_root / cfg.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, cfg.level.upper(), logging.INFO)

    # Stdlib logger setup — structlog wraps this.
    handlers: list[logging.Handler] = []

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    handlers.append(file_handler)

    if cfg.log_to_stdout:
        stdout_handler = logging.StreamHandler(sys.stderr)
        stdout_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(stdout_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Quiet noisy third parties.
    for noisy in ("urllib3", "httpx", "httpcore", "asyncio", "ctranslate2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    timestamper = structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]

    if cfg.renderer == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True, exception_formatter=structlog.dev.plain_traceback)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)

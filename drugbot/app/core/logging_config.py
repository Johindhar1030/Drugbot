"""Centralized logging configuration for DrugBot.

Provides:
1. Application file logging (logs/drugbot.log) with rotation.
2. Audit file logging (logs/audit.log) with rotation.
3. Concurrent console logging for development.
4. Sensitive data masking (passwords, tokens, API keys).
5. Duplicate handler prevention.
"""

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from app.core.config import settings

# Sensitive patterns to redact from all log output
SENSITIVE_PATTERNS = [
    (re.compile(r"gsk_[A-Za-z0-9]{20,}", re.IGNORECASE), "[REDACTED_GROQ_KEY]"),
    (re.compile(r"ck-[A-Za-z0-9]{20,}", re.IGNORECASE), "[REDACTED_CHROMA_KEY]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9\._\-]+", re.IGNORECASE), r"\1[REDACTED_TOKEN]"),
    (re.compile(r"(\bpassword\b\s*[:=]\s*)[^\s,;&'\"]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(\btoken\b\s*[:=]\s*)[^\s,;&'\"]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(\bsecret\b\s*[:=]\s*)[^\s,;&'\"]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(\bapi[_-]?key\b\s*[:=]\s*)[^\s,;&'\"]+", re.IGNORECASE), r"\1[REDACTED]"),
]


class SensitiveDataFormatter(logging.Formatter):
    """Log formatter that automatically masks sensitive tokens, keys, and credentials."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        sanitized = original
        for pattern, replacement in SENSITIVE_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized


def setup_logging(
    log_dir: str | Path | None = None,
    log_level: str | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> tuple[Path, Path]:
    """Configure centralized application and audit logging handlers.

    Returns a tuple of (drugbot_log_path, audit_log_path).
    """
    # 1. Resolve configuration values
    resolved_dir = Path(log_dir or getattr(settings, "log_dir", "logs"))
    resolved_dir.mkdir(parents=True, exist_ok=True)

    level_str = (log_level or getattr(settings, "log_level", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)

    max_b = max_bytes if max_bytes is not None else getattr(settings, "log_max_bytes", 10485760)
    b_count = backup_count if backup_count is not None else getattr(settings, "log_backup_count", 5)

    app_log_path = resolved_dir / "drugbot.log"
    audit_log_path = resolved_dir / "audit.log"

    # Common formatters
    app_formatter = SensitiveDataFormatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    audit_formatter = SensitiveDataFormatter(
        "%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 2. Configure Root / Application Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove pre-existing handlers on root to prevent duplicate logs
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    # File handler for drugbot.log
    app_file_handler = RotatingFileHandler(
        app_log_path, maxBytes=max_b, backupCount=b_count, encoding="utf-8"
    )
    app_file_handler.setLevel(level)
    app_file_handler.setFormatter(app_formatter)
    root_logger.addHandler(app_file_handler)

    # Console handler for stdout/stderr
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(app_formatter)
    root_logger.addHandler(console_handler)

    # 3. Configure Dedicated Audit Logger
    audit_logger = logging.getLogger("drugbot.audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # Prevent audit events from propagating to drugbot.log

    # Remove pre-existing handlers on audit_logger to prevent duplicates
    for h in list(audit_logger.handlers):
        audit_logger.removeHandler(h)

    # File handler for audit.log
    audit_file_handler = RotatingFileHandler(
        audit_log_path, maxBytes=max_b, backupCount=b_count, encoding="utf-8"
    )
    audit_file_handler.setLevel(logging.INFO)
    audit_file_handler.setFormatter(audit_formatter)
    audit_logger.addHandler(audit_file_handler)

    # Console handler for audit events
    audit_console_handler = logging.StreamHandler()
    audit_console_handler.setLevel(logging.INFO)
    audit_console_handler.setFormatter(audit_formatter)
    audit_logger.addHandler(audit_console_handler)

    return app_log_path, audit_log_path

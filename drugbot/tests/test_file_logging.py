"""Test suite for DrugBot separate file-based logging system.

Verifies:
1. Automatically creates the logs directory.
2. Creates drugbot.log and audit.log files.
3. App logs route to drugbot.log.
4. Audit events route to audit.log without propagating to drugbot.log.
5. Console logging handlers are attached.
6. Prevents duplicate handlers when setup_logging() is invoked repeatedly.
7. Configures RotatingFileHandler with max_bytes and backup_count.
8. Masks sensitive values (passwords, tokens, API keys).
"""

import logging
import tempfile
from pathlib import Path
import pytest

from app.core.logging_config import setup_logging, SensitiveDataFormatter
from app.auth.security import log_audit_event


def test_logs_directory_and_files_creation(tmp_path):
    log_dir = tmp_path / "logs"
    assert not log_dir.exists()

    app_log, audit_log = setup_logging(log_dir=log_dir)

    assert log_dir.exists()
    assert log_dir.is_dir()
    assert app_log == log_dir / "drugbot.log"
    assert audit_log == log_dir / "audit.log"


def test_app_and_audit_logs_routing(tmp_path):
    log_dir = tmp_path / "test_logs"
    app_log, audit_log = setup_logging(log_dir=log_dir)

    test_app_logger = logging.getLogger("app.test_module")
    test_app_logger.info("Normal application event occurred")

    log_audit_event(
        action="TEST_ACTION",
        resource_type="TEST_RESOURCE",
        user_id=42,
        status="SUCCESS",
        details="Test details",
    )

    # Force handlers to flush content to disk
    for handler in logging.getLogger().handlers:
        handler.flush()
    for handler in logging.getLogger("drugbot.audit").handlers:
        handler.flush()

    app_content = app_log.read_text(encoding="utf-8")
    audit_content = audit_log.read_text(encoding="utf-8")

    # App log must contain normal application log
    assert "Normal application event occurred" in app_content

    # Audit log must contain audit event
    assert "AUDIT_EVENT" in audit_content
    assert "action=TEST_ACTION" in audit_content
    assert "user_id=42" in audit_content

    # Audit events must NOT propagate to drugbot.log (no duplicate audit entries in app log)
    assert "action=TEST_ACTION" not in app_content


def test_console_handler_attached(tmp_path):
    log_dir = tmp_path / "console_logs"
    setup_logging(log_dir=log_dir)

    root_handlers = logging.getLogger().handlers
    audit_handlers = logging.getLogger("drugbot.audit").handlers

    has_console = any(isinstance(h, logging.StreamHandler) and not hasattr(h, "baseFilename") for h in root_handlers)
    has_audit_console = any(isinstance(h, logging.StreamHandler) and not hasattr(h, "baseFilename") for h in audit_handlers)

    assert has_console is True
    assert has_audit_console is True


def test_no_duplicate_handlers_on_reinit(tmp_path):
    log_dir = tmp_path / "reinit_logs"
    setup_logging(log_dir=log_dir)
    initial_root_count = len(logging.getLogger().handlers)
    initial_audit_count = len(logging.getLogger("drugbot.audit").handlers)

    # Call setup_logging repeatedly
    setup_logging(log_dir=log_dir)
    setup_logging(log_dir=log_dir)

    assert len(logging.getLogger().handlers) == initial_root_count
    assert len(logging.getLogger("drugbot.audit").handlers) == initial_audit_count


def test_log_rotation_configuration(tmp_path):
    log_dir = tmp_path / "rotation_logs"
    max_bytes = 1000
    backup_count = 3
    setup_logging(log_dir=log_dir, max_bytes=max_bytes, backup_count=backup_count)

    root_file_handler = next(h for h in logging.getLogger().handlers if hasattr(h, "baseFilename"))
    audit_file_handler = next(h for h in logging.getLogger("drugbot.audit").handlers if hasattr(h, "baseFilename"))

    assert root_file_handler.maxBytes == max_bytes
    assert root_file_handler.backupCount == backup_count
    assert audit_file_handler.maxBytes == max_bytes
    assert audit_file_handler.backupCount == backup_count


def test_sensitive_data_redaction():
    formatter = SensitiveDataFormatter("%(message)s")

    record_groq = logging.LogRecord("test", logging.INFO, "", 0, "Connecting with GROQ_API_KEY=gsk_1234567890abcdef1234567890abcdef", (), None)
    record_chroma = logging.LogRecord("test", logging.INFO, "", 0, "Chroma key: ck-1234567890abcdef1234567890abcdef", (), None)
    record_token = logging.LogRecord("test", logging.INFO, "", 0, "Header: Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMiJ9", (), None)
    record_password = logging.LogRecord("test", logging.INFO, "", 0, "Payload password=SecretPass123!", (), None)

    assert "gsk_" not in formatter.format(record_groq)
    assert "[REDACTED_GROQ_KEY]" in formatter.format(record_groq)

    assert "ck-" not in formatter.format(record_chroma)
    assert "[REDACTED_CHROMA_KEY]" in formatter.format(record_chroma)

    assert "eyJhbGci" not in formatter.format(record_token)
    assert "[REDACTED_TOKEN]" in formatter.format(record_token)

    assert "SecretPass123!" not in formatter.format(record_password)
    assert "[REDACTED]" in formatter.format(record_password)

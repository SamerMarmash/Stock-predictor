"""Tests for configuration module."""

import os

from src.core.config import Settings


def test_settings_defaults():
    settings = Settings(
        _env_file=None,  # Don't load .env in tests
    )
    assert settings.prediction_interval_minutes == 60
    assert settings.default_confidence_threshold == 60
    assert settings.llm_provider == "anthropic"
    assert settings.log_level == "INFO"


def test_notification_url_list():
    settings = Settings(
        notification_urls="discord://a/b/c, slack://x/y/z",
        _env_file=None,
    )
    urls = settings.notification_url_list
    assert len(urls) == 2
    assert urls[0] == "discord://a/b/c"
    assert urls[1] == "slack://x/y/z"


def test_empty_notification_urls():
    settings = Settings(notification_urls="", _env_file=None)
    assert settings.notification_url_list == []


def test_active_api_key_anthropic():
    settings = Settings(
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-test",
        _env_file=None,
    )
    assert settings.active_api_key == "sk-ant-test"


def test_active_api_key_openai():
    settings = Settings(
        llm_provider="openai",
        openai_api_key="sk-test",
        _env_file=None,
    )
    assert settings.active_api_key == "sk-test"

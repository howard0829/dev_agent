"""
tests/test_config.py — config.py 환경변수 로딩 테스트
"""

import importlib
import os


def test_default_values():
    """기본값이 올바르게 설정되는지 확인"""
    import config

    assert config.OLLAMA_DEFAULT_URL == os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    assert config.PROXY_PORT == int(os.getenv("PROXY_PORT", "8082"))
    assert config.AGENT_MAX_TURNS == int(os.getenv("AGENT_MAX_TURNS", "150"))
    assert config.MAX_FILE_SIZE_MB == int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    assert config.WORKSPACE_EXPIRE_HOURS == int(os.getenv("WORKSPACE_EXPIRE_HOURS", "24"))
    assert isinstance(config.CORS_ORIGINS, list)
    assert isinstance(config.ALLOWED_EXTENSIONS, set)
    assert ".py" in config.ALLOWED_EXTENSIONS
    assert ".md" in config.ALLOWED_EXTENSIONS


def test_environment_override(monkeypatch):
    """환경변수 오버라이드가 반영되는지 확인"""
    monkeypatch.setenv("PROXY_PORT", "9999")
    monkeypatch.setenv("AGENT_MAX_TURNS", "50")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:11434")

    import config
    importlib.reload(config)

    assert config.PROXY_PORT == 9999
    assert config.AGENT_MAX_TURNS == 50
    assert config.OLLAMA_DEFAULT_URL == "http://custom:11434"


def test_cors_origins_parsing(monkeypatch):
    """CORS_ORIGINS 쉼표 구분 파싱"""
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com , http://b.com, http://c.com")

    import config
    importlib.reload(config)

    assert config.CORS_ORIGINS == ["http://a.com", "http://b.com", "http://c.com"]


def test_code_embedding_model_default():
    """CODE_EMBEDDING_MODEL 기본값은 빈 문자열"""
    import config
    assert isinstance(config.CODE_EMBEDDING_MODEL, str)

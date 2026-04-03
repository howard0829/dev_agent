"""
tests/conftest.py — 공통 fixture
"""

import os
import tempfile
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_workspace(tmp_path):
    """임시 워크스페이스 디렉토리"""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def mock_session_state(monkeypatch):
    """Streamlit session_state를 일반 dict로 모킹"""
    state = {}

    class FakeSessionState(dict):
        pass

    fake = FakeSessionState(state)
    monkeypatch.setattr("streamlit.session_state", fake)
    return fake

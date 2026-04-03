"""
tests/test_session.py — core/session.py 네임스페이스 세션 상태 테스트
"""

from core.session import ns, get_state, set_state, display_path


def test_ns_generates_namespaced_key():
    """ns()가 prefix.key 형식의 키를 생성하는지 확인"""
    assert ns("deep_assist", "messages") == "deep_assist.messages"
    assert ns("test_mancer", "is_running") == "test_mancer.is_running"
    assert ns("app", "key") == "app.key"


def test_get_set_state(mock_session_state):
    """get_state/set_state가 네임스페이스 세션에 올바르게 읽고 쓰는지 확인"""
    set_state("myapp", "counter", 42)
    assert get_state("myapp", "counter") == 42
    assert mock_session_state["myapp.counter"] == 42


def test_get_state_default(mock_session_state):
    """존재하지 않는 키에 대해 기본값을 반환하는지 확인"""
    assert get_state("myapp", "nonexistent") is None
    assert get_state("myapp", "nonexistent", "default") == "default"


def test_set_state_overwrite(mock_session_state):
    """set_state가 기존 값을 덮어쓰는지 확인"""
    set_state("app", "val", "old")
    set_state("app", "val", "new")
    assert get_state("app", "val") == "new"


def test_display_path_with_workspaces():
    """display_path가 /workspaces/ 이후만 추출하는지 확인"""
    path = "/home/user/data/workspaces/abc123/notes.md"
    assert display_path(path) == "/workspaces/abc123/notes.md"


def test_display_path_without_workspaces():
    """display_path가 /workspaces/가 없으면 전체 경로를 반환하는지 확인"""
    path = "/home/user/project/main.py"
    assert display_path(path) == path

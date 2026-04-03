"""
tests/test_server.py — FastAPI 파일 서버 엔드포인트 테스트
"""

import os
import pytest


@pytest.fixture
def test_client(tmp_workspace, monkeypatch):
    """FastAPI TestClient 생성 (임시 워크스페이스 사용)"""
    monkeypatch.setenv("WORKSPACES_ROOT", str(tmp_workspace.parent))

    # server.py 재임포트를 위해 config 갱신
    import importlib
    import config
    importlib.reload(config)

    import server
    importlib.reload(server)

    from fastapi.testclient import TestClient
    return TestClient(server.app)


def test_health_endpoint(test_client):
    """GET /api/health 200 응답"""
    resp = test_client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_session_endpoint(test_client):
    """GET /api/session 세션 정보 반환"""
    resp = test_client.get("/api/session")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "workspace_path" in data


def test_file_upload_and_read(test_client):
    """파일 업로드 후 읽기"""
    content = b"hello world"
    resp = test_client.post(
        "/api/files/upload",
        files={"file": ("test.txt", content, "text/plain")},
    )
    assert resp.status_code == 200

    # 읽기
    resp = test_client.get("/api/files/read/test.txt")
    assert resp.status_code == 200
    assert resp.json()["content"] == "hello world"


def test_file_write_and_download(test_client):
    """파일 작성 후 다운로드"""
    resp = test_client.post(
        "/api/files/write",
        json={"path": "notes.md", "content": "# Hello"},
    )
    assert resp.status_code == 200

    resp = test_client.get("/api/files/download/notes.md")
    assert resp.status_code == 200
    assert b"# Hello" in resp.content


def test_file_delete(test_client):
    """파일 삭제"""
    # 생성
    test_client.post("/api/files/write", json={"path": "delete_me.txt", "content": "x"})
    # 삭제
    resp = test_client.delete("/api/files/delete_me.txt")
    assert resp.status_code == 200

    # 다시 읽기 시도 → 404
    resp = test_client.get("/api/files/read/delete_me.txt")
    assert resp.status_code == 404


def test_path_traversal_blocked(test_client):
    """Path Traversal 공격 차단"""
    resp = test_client.post(
        "/api/files/write",
        json={"path": "../../../etc/passwd", "content": "malicious"},
    )
    assert resp.status_code in (400, 403)


def test_disallowed_extension(test_client):
    """허용되지 않은 확장자 차단"""
    resp = test_client.post(
        "/api/files/upload",
        files={"file": ("malware.exe", b"bad", "application/octet-stream")},
    )
    assert resp.status_code in (400, 403)


def test_file_list(test_client):
    """파일 목록 조회"""
    test_client.post("/api/files/write", json={"path": "a.txt", "content": "a"})
    test_client.post("/api/files/write", json={"path": "b.py", "content": "b"})

    resp = test_client.get("/api/files/listdir", params={"path": ""})
    assert resp.status_code == 200
    items = resp.json().get("items", [])
    names = [i["name"] for i in items]
    assert "a.txt" in names
    assert "b.py" in names

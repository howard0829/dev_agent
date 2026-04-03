"""
DeepAssist - FastAPI 파일/워크스페이스 관리 서버

역할:
  - 클라이언트 IP 기반으로 워크스페이스 폴더를 자동 생성/관리
  - 파일 업로드 / 다운로드 / 목록 조회 REST API 제공
  - 비활성 워크스페이스 자동 정리 (만료 시간 경과 시 삭제)
  - 워크스페이스별 디스크 용량 제한 (쿼터)
  - Streamlit(app.py)이 httpx를 통해 이 서버와 통신

에이전트 실행은 app.py(Streamlit)에서 직접 처리합니다.
"""

import logging
import os
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from config import (
    WORKSPACES_ROOT as _WORKSPACES_ROOT_STR,
    MAX_FILE_SIZE_MB, MAX_WORKSPACE_SIZE_MB,
    WORKSPACE_EXPIRE_HOURS, CLEANUP_INTERVAL_MINUTES,
    ALLOWED_EXTENSIONS, CORS_ORIGINS,
    FILE_SERVER_PORT,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 설정 (config.py에서 로드)
# ──────────────────────────────────────────────
WORKSPACES_ROOT = Path(_WORKSPACES_ROOT_STR)


# ──────────────────────────────────────────────
# 워크스페이스 자동 정리 (Lifespan)
# ──────────────────────────────────────────────

def cleanup_expired_workspaces():
    """만료 시간이 지난 워크스페이스 하위 폴더를 자동 삭제"""
    if not WORKSPACES_ROOT.exists():
        return

    now = datetime.now()
    expire_delta = timedelta(hours=WORKSPACE_EXPIRE_HOURS)
    deleted_count = 0

    for folder in WORKSPACES_ROOT.iterdir():
        if not folder.is_dir():
            continue
        # 폴더 수정 시간 기준으로 만료 여부 판단
        mtime = datetime.fromtimestamp(folder.stat().st_mtime)
        if now - mtime > expire_delta:
            try:
                shutil.rmtree(folder)
                deleted_count += 1
                logger.info(f"🗑️ 만료된 워크스페이스 삭제: {folder.name} (마지막 수정: {mtime.strftime('%Y-%m-%d %H:%M')})")
            except Exception as e:
                logger.warning(f"⚠️ 워크스페이스 삭제 실패: {folder.name} - {e}")

    if deleted_count > 0:
        logger.info(f"✅ 총 {deleted_count}개의 만료 워크스페이스 정리 완료")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 만료 워크스페이스 정리 + 주기적 정리 태스크 가동"""
    import asyncio

    WORKSPACES_ROOT.mkdir(exist_ok=True)
    cleanup_expired_workspaces()  # 시작 시 즉시 한 번 실행

    async def periodic_cleanup():
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL_MINUTES * 60)
            cleanup_expired_workspaces()

    task = asyncio.create_task(periodic_cleanup())
    yield
    task.cancel()


app = FastAPI(
    title="DeepAssist File Server",
    description="워크스페이스 및 파일 관리 API",
    version="1.1.0",
    lifespan=lifespan
)

# CORS: Streamlit에서 FastAPI로의 요청 허용 (환경변수로 설정 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def get_client_ip(request: Request) -> str:
    """Nginx 리버스 프록시 환경을 고려하여 클라이언트 실제 IP 추출"""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host or "unknown"


def get_session_id(request: Request) -> str:
    """IP + User-Agent 기반으로 세션 ID 생성 (짧은 해시)"""
    ip = get_client_ip(request)
    ua = request.headers.get("User-Agent", "")
    raw = f"{ip}:{ua}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def get_workspace(session_id: str) -> Path:
    """세션 전용 워크스페이스 경로 반환 (없으면 자동 생성)"""
    workspace = WORKSPACES_ROOT / session_id
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def get_workspace_size(workspace: Path) -> int:
    """워크스페이스 폴더의 총 파일 크기(bytes)를 재귀적으로 계산"""
    total = 0
    for f in workspace.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def is_safe_path(workspace: Path, target: Path) -> bool:
    """경로 탈출(Path Traversal) 공격 방지: workspace 하위인지 검증.

    symlink를 해소(resolve)한 뒤 실제 경로가 workspace 내부인지 확인한다.
    """
    try:
        resolved_target = target.resolve()
        resolved_workspace = workspace.resolve()
        resolved_target.relative_to(resolved_workspace)
        # symlink가 workspace 외부를 가리키면 차단
        if target.is_symlink():
            link_target = target.resolve()
            link_target.relative_to(resolved_workspace)
        return True
    except ValueError:
        return False


def touch_workspace(workspace: Path):
    """워크스페이스 폴더의 수정 시간을 갱신하여 만료 타이머를 리셋"""
    os.utime(workspace, None)


def _format_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ──────────────────────────────────────────────
# API 엔드포인트
# ──────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root_redirect():
    """루트 접속 시 API 문서로 자동 리다이렉트"""
    return RedirectResponse(url="/docs")


@app.get("/api/session", summary="현재 세션 정보 조회")
async def get_session_info(request: Request):
    """세션 ID와 워크스페이스 경로를 반환합니다."""
    session_id = get_session_id(request)
    workspace = get_workspace(session_id)
    ws_size = get_workspace_size(workspace)
    return {
        "session_id": session_id,
        "client_ip": get_client_ip(request),
        "workspace_path": str(workspace.resolve()),
        "workspace_size": _format_size(ws_size),
        "workspace_quota": f"{MAX_WORKSPACE_SIZE_MB} MB",
    }


@app.get("/api/files/list", summary="워크스페이스 파일 목록 조회")
async def list_files(request: Request):
    """현재 세션의 워크스페이스에 있는 파일 목록과 메타데이터를 반환합니다."""
    session_id = get_session_id(request)
    workspace = get_workspace(session_id)
    touch_workspace(workspace)

    files = []
    for item in sorted(workspace.iterdir()):
        if item.is_file():
            stat = item.stat()
            files.append({
                "name": item.name,
                "size_bytes": stat.st_size,
                "size_display": _format_size(stat.st_size),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "extension": item.suffix.lower(),
            })

    ws_size = get_workspace_size(workspace)
    return {
        "session_id": session_id,
        "files": files,
        "count": len(files),
        "workspace_size": _format_size(ws_size),
        "workspace_quota": f"{MAX_WORKSPACE_SIZE_MB} MB",
    }


@app.get("/api/files/listdir", summary="특정 디렉토리 직접 자식 목록 조회")
async def listdir(request: Request, path: str = ""):
    """워크스페이스 내 특정 폴더의 직계 자식(폴더/파일)을 반환합니다. path가 비어있으면 루트."""
    session_id = get_session_id(request)
    workspace = get_workspace(session_id)
    target = (workspace / path) if path else workspace

    if not is_safe_path(workspace, target):
        raise HTTPException(status_code=403, detail="접근이 거부되었습니다.")
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail=f"디렉토리를 찾을 수 없습니다: {path}")

    items = []
    for item in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        rel = str(item.relative_to(workspace))
        if item.is_dir():
            items.append({"name": item.name, "path": rel, "type": "directory"})
        else:
            stat = item.stat()
            items.append({
                "name": item.name,
                "path": rel,
                "type": "file",
                "size": _format_size(stat.st_size),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })

    ws_size = get_workspace_size(workspace)
    return {
        "path": path,
        "items": items,
        "workspace_size": _format_size(ws_size),
        "workspace_quota": f"{MAX_WORKSPACE_SIZE_MB} MB",
    }


@app.post("/api/files/upload", summary="파일 업로드")
async def upload_file(request: Request, file: UploadFile = File(...), folder: str = ""):
    """워크스페이스에 파일을 업로드합니다. folder 파라미터로 업로드 대상 하위 폴더를 지정할 수 있습니다."""
    session_id = get_session_id(request)
    workspace = get_workspace(session_id)

    # 대상 디렉토리 결정
    target_dir = (workspace / folder) if folder else workspace
    if not is_safe_path(workspace, target_dir):
        raise HTTPException(status_code=403, detail="접근이 거부되었습니다.")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 확장자 검증
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"허용되지 않는 파일 형식입니다: {ext}. 허용 목록: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 파일 크기 검증
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"파일 크기가 {MAX_FILE_SIZE_MB}MB를 초과합니다."
        )

    # 워크스페이스 쿼터 검증
    current_size = get_workspace_size(workspace)
    if current_size + len(content) > MAX_WORKSPACE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"워크스페이스 용량 한도({MAX_WORKSPACE_SIZE_MB}MB)를 초과합니다. "
                   f"현재 사용량: {_format_size(current_size)}"
        )

    safe_name = Path(file.filename).name
    target_path = target_dir / safe_name

    with open(target_path, "wb") as f:
        f.write(content)

    touch_workspace(workspace)
    return {
        "message": "업로드 성공",
        "filename": safe_name,
        "size_bytes": len(content),
        "workspace_path": str(target_path.resolve()),
    }


@app.get("/api/files/download/{filename}", summary="파일 다운로드")
async def download_file(filename: str, request: Request):
    """워크스페이스의 특정 파일을 다운로드합니다."""
    session_id = get_session_id(request)
    workspace = get_workspace(session_id)

    target_path = workspace / filename

    if not is_safe_path(workspace, target_path):
        raise HTTPException(status_code=403, detail="접근이 거부되었습니다.")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {filename}")

    touch_workspace(workspace)
    return FileResponse(
        path=str(target_path),
        filename=filename,
        media_type="application/octet-stream"
    )


@app.delete("/api/files/{filename}", summary="파일 삭제")
async def delete_file(filename: str, request: Request):
    """워크스페이스의 특정 파일을 삭제합니다."""
    session_id = get_session_id(request)
    workspace = get_workspace(session_id)
    target_path = workspace / filename

    if not is_safe_path(workspace, target_path):
        raise HTTPException(status_code=403, detail="접근이 거부되었습니다.")

    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {filename}")

    target_path.unlink()
    touch_workspace(workspace)
    return {"message": f"'{filename}' 삭제 완료"}


@app.get("/api/files/read/{file_path:path}", summary="파일 내용 읽기")
async def read_file_content(file_path: str, request: Request):
    """워크스페이스 내 파일의 텍스트 내용을 반환합니다."""
    session_id = get_session_id(request)
    workspace = get_workspace(session_id)
    target = workspace / file_path

    if not is_safe_path(workspace, target):
        raise HTTPException(status_code=403, detail="접근이 거부되었습니다.")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {file_path}")

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="바이너리 파일은 편집할 수 없습니다.")

    return {"path": file_path, "content": content, "size_bytes": target.stat().st_size}


from pydantic import BaseModel

class FileWriteRequest(BaseModel):
    path: str
    content: str

@app.post("/api/files/write", summary="파일 내용 저장")
async def write_file_content(body: FileWriteRequest, request: Request):
    """워크스페이스 내 파일에 텍스트 내용을 저장합니다."""
    session_id = get_session_id(request)
    workspace = get_workspace(session_id)
    target = workspace / body.path

    if not is_safe_path(workspace, target):
        raise HTTPException(status_code=403, detail="접근이 거부되었습니다.")

    # 쿼터 검증
    new_size = len(body.content.encode("utf-8"))
    existing_size = target.stat().st_size if target.exists() else 0
    ws_size = get_workspace_size(workspace) - existing_size + new_size
    if ws_size > MAX_WORKSPACE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"워크스페이스 용량 한도({MAX_WORKSPACE_SIZE_MB}MB)를 초과합니다.")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body.content, encoding="utf-8")
    touch_workspace(workspace)
    return {"message": f"'{body.path}' 저장 완료", "size_bytes": new_size}


@app.get("/api/health", summary="서버 상태 확인")
async def health_check():
    return {"status": "ok", "service": "DeepAssist File Server"}


# ──────────────────────────────────────────────
# 직접 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    WORKSPACES_ROOT.mkdir(exist_ok=True)
    logger.info("🚀 DeepAssist File Server 시작 중...")
    logger.info(f"   워크스페이스 루트: {WORKSPACES_ROOT.resolve()}")
    logger.info(f"   워크스페이스 쿼터: {MAX_WORKSPACE_SIZE_MB}MB / 만료: {WORKSPACE_EXPIRE_HOURS}시간")
    logger.info(f"   API 문서: http://localhost:{FILE_SERVER_PORT}/docs")
    uvicorn.run("server:app", host="0.0.0.0", port=FILE_SERVER_PORT, reload=True)

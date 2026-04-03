"""
중앙 설정 모듈 — 환경변수 기반 프로젝트 전역 설정

모든 하드코딩된 URL, 포트, 상수를 환경변수로 통합 관리합니다.
.env 파일에서 자동 로드됩니다.
"""

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# LLM 프로바이더 기본값
# ──────────────────────────────────────────────

OLLAMA_DEFAULT_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL: str = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen3:8b")
VLLM_DEFAULT_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_DEFAULT_MODEL: str = os.getenv("VLLM_DEFAULT_MODEL", "Qwen/Qwen3-8B")
GEMINI_DEFAULT_MODEL: str = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash-lite")

# ──────────────────────────────────────────────
# 임베딩 설정
# ──────────────────────────────────────────────

EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest")
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
CODE_EMBEDDING_MODEL: str = os.getenv("CODE_EMBEDDING_MODEL", "")

# ──────────────────────────────────────────────
# OpenAI 호환 직접 연결 설정
# ──────────────────────────────────────────────

OPENAI_DIRECT_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "http://localhost:8082")
OPENAI_DIRECT_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_DIRECT_MODEL: str = os.getenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "")

# ──────────────────────────────────────────────
# claude-code-proxy 설정
# ──────────────────────────────────────────────

PROXY_PORT: int = int(os.getenv("PROXY_PORT", "8082"))
PROXY_MAX_WAIT: int = int(os.getenv("PROXY_MAX_WAIT", "15"))

# ──────────────────────────────────────────────
# 파일 서버 설정
# ──────────────────────────────────────────────

FILE_SERVER_URL: str = os.getenv("FILE_SERVER_URL", "http://localhost:8000")
FILE_SERVER_PORT: int = int(os.getenv("FILE_SERVER_PORT", "8000"))

# ──────────────────────────────────────────────
# 워크스페이스 설정
# ──────────────────────────────────────────────

WORKSPACES_ROOT: str = os.getenv("WORKSPACES_ROOT", "./workspaces")
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
MAX_WORKSPACE_SIZE_MB: int = int(os.getenv("MAX_WORKSPACE_SIZE_MB", "100"))
WORKSPACE_EXPIRE_HOURS: int = int(os.getenv("WORKSPACE_EXPIRE_HOURS", "24"))
CLEANUP_INTERVAL_MINUTES: int = int(os.getenv("CLEANUP_INTERVAL_MINUTES", "30"))
ALLOWED_EXTENSIONS: set = {
    ".md", ".txt", ".py", ".json", ".yaml", ".yml",
    ".csv", ".html", ".js", ".ts", ".sh", ".env",
    ".toml", ".ini", ".cfg", ".log",
}

# ──────────────────────────────────────────────
# CORS 설정
# ──────────────────────────────────────────────

CORS_ORIGINS: List[str] = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
]

# ──────────────────────────────────────────────
# 에이전트 설정
# ──────────────────────────────────────────────

AGENT_MAX_TURNS: int = int(os.getenv("AGENT_MAX_TURNS", "150"))
DEEPASSIST_MD_MAX_SIZE: int = int(os.getenv("DEEPASSIST_MD_MAX_SIZE", "50000"))

# ──────────────────────────────────────────────
# Knowledge DB 설정
# ──────────────────────────────────────────────

KNOWLEDGE_BASE_DIR: str = os.path.expanduser(
    os.getenv("KNOWLEDGE_BASE_DIR", "~/.deepassist/knowledge")
)

"""
백엔드 전략 모듈 — Claude Agent SDK의 백엔드 연결 전략 패턴
claude-code-proxy 경유, Ollama Native Anthropic API, vLLM 3종을 지원한다.
"""

import logging
import os
import re
import time
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Optional

import requests

from config import (
    OLLAMA_DEFAULT_URL, VLLM_DEFAULT_URL,
    PROXY_PORT, PROXY_MAX_WAIT,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 추상 기반 클래스
# ──────────────────────────────────────────────

class BackendStrategy(ABC):
    """Claude Agent SDK의 백엔드 연결 전략 인터페이스.

    activate()에서 환경변수를 설정하고, 필요 시 프록시 프로세스를 기동한다.
    cleanup()에서 프로세스를 종료하고, 로그를 분석하여 드롭된 파라미터를 보고한다.
    """

    @abstractmethod
    def check(self) -> tuple[bool, str]:
        """연결 가능 여부를 사전 확인한다."""

    @abstractmethod
    def activate(self, event_queue) -> None:
        """환경변수를 설정하고, 필요 시 프록시 프로세스를 시작한다."""

    @abstractmethod
    def cleanup(self, event_queue) -> None:
        """프로세스를 종료하고, 리소스를 정리한다."""


# ──────────────────────────────────────────────
# 1) claude-code-proxy 경유 (Ollama/Gemini)
# ──────────────────────────────────────────────

# config.py에서 가져옴
_PROXY_MAX_WAIT = PROXY_MAX_WAIT
_PROXY_PORT = PROXY_PORT


class ProxyStrategy(BackendStrategy):
    """claude-code-proxy를 경유하는 전략.

    Claude API 전용 프록시(claude-code-proxy)를 로컬에서 기동하여
    Claude Agent SDK의 요청을 Ollama/Gemini로 라우팅한다.
    Claude 특수 파라미터(context_management 등) 처리와 tool calling 변환을
    프록시가 자체적으로 처리하므로 별도 워크어라운드가 필요 없다.
    """

    def __init__(
        self,
        llm_provider: str,
        model: str,
        api_key: str = "",
        ollama_url: str = "",
    ):
        self.llm_provider = llm_provider
        self.model = model
        self.api_key = api_key
        self.ollama_url = ollama_url or OLLAMA_DEFAULT_URL

        self._proxy_process: Optional[subprocess.Popen] = None
        self._log_path: Optional[str] = None
        self._log_file = None

    # ── check ──

    def check(self) -> tuple[bool, str]:
        import shutil

        if not shutil.which("claude-code-proxy"):
            return False, (
                "claude-code-proxy CLI를 찾을 수 없습니다. "
                "'pip install claude-code-proxy'로 설치하세요."
            )

        # Ollama: 서버 접근성 확인
        if self.llm_provider == "Ollama":
            ok, msg = _check_ollama(self.ollama_url, self.model)
            if not ok:
                return False, msg

        return True, f"Proxy 모드 준비됨 ({self.llm_provider}/{self.model})"

    # ── activate ──

    def activate(self, event_queue) -> None:
        import shutil

        event_queue.put(("status", f"[Proxy] {self.llm_provider} 연결을 위한 claude-code-proxy 시작 중..."))

        # 프로바이더별 환경변수 설정 (claude-code-proxy가 읽음)
        proxy_env = os.environ.copy()
        proxy_env["PORT"] = str(_PROXY_PORT)

        if self.llm_provider == "Ollama":
            proxy_env["PREFERRED_PROVIDER"] = "ollama"
            proxy_env["BIG_MODEL"] = self.model
            proxy_env["SMALL_MODEL"] = self.model
            proxy_env["OLLAMA_API_BASE"] = self.ollama_url
        elif self.llm_provider in ("Gemini API", "Gemini"):
            proxy_env["PREFERRED_PROVIDER"] = "google"
            proxy_env["BIG_MODEL"] = self.model
            proxy_env["SMALL_MODEL"] = self.model
            proxy_env["GEMINI_API_KEY"] = self.api_key

        # SDK가 프록시를 바라보도록 환경변수 설정
        os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{_PROXY_PORT}"
        os.environ["ANTHROPIC_API_KEY"] = self.api_key if self.api_key else "dummy"

        # 로그 파일
        log_tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", prefix="proxy_log_", delete=False
        )
        self._log_path = log_tmp.name
        log_tmp.close()
        self._log_file = open(self._log_path, "w", encoding="utf-8")

        # claude-code-proxy 프로세스 시작
        proxy_path = shutil.which("claude-code-proxy") or "claude-code-proxy"
        self._proxy_process = subprocess.Popen(
            [proxy_path],
            env=proxy_env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )

        # Health check 폴링
        self._wait_for_health(event_queue)

    # ── cleanup ──

    def cleanup(self, event_queue) -> None:
        if self._proxy_process:
            self._proxy_process.terminate()
            try:
                self._proxy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proxy_process.kill()
                self._proxy_process.wait(timeout=3)
            finally:
                self._proxy_process = None

        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

        _safe_unlink(self._log_path)

    # ── 내부 헬퍼 ──

    def _wait_for_health(self, event_queue) -> None:
        """프록시 기동 대기 + Health check 폴링."""
        started = False
        for i in range(_PROXY_MAX_WAIT):
            time.sleep(1)
            if self._proxy_process and self._proxy_process.poll() is not None:
                if self._log_file:
                    self._log_file.close()
                log_tail = _read_log_tail(self._log_path, 1000)
                raise RuntimeError(
                    f"claude-code-proxy가 시작 직후 종료되었습니다 "
                    f"(exit code: {self._proxy_process.returncode}).\n"
                    f"로그:\n{log_tail}"
                )
            try:
                resp = requests.get(f"http://127.0.0.1:{_PROXY_PORT}/health", timeout=2)
                if resp.status_code == 200:
                    started = True
                    event_queue.put(("status", f"[Proxy] claude-code-proxy 시작 완료 ({i + 1}초)"))
                    break
            except requests.ConnectionError:
                pass
            except Exception:
                pass

        if not started:
            if self._proxy_process and self._proxy_process.poll() is not None:
                if self._log_file:
                    self._log_file.close()
                log_tail = _read_log_tail(self._log_path, 1000)
                raise RuntimeError(
                    f"claude-code-proxy 시작 실패 "
                    f"(exit code: {self._proxy_process.returncode}).\n"
                    f"로그:\n{log_tail}"
                )
            event_queue.put(("status",
                f"⚠️ claude-code-proxy Health check 실패 ({_PROXY_MAX_WAIT}초 대기). 계속 진행합니다..."))


# ──────────────────────────────────────────────
# 2) Ollama Native Anthropic API (실험적)
# ──────────────────────────────────────────────

class OllamaNativeStrategy(BackendStrategy):
    """Ollama의 네이티브 Anthropic API 호환 모드를 사용하는 전략.

    프록시 없이 Ollama(v0.14+)에 직접 연결한다.
    Ollama가 /v1/messages 엔드포인트에서 Anthropic Messages API를 직접 처리한다.
    """

    def __init__(self, model: str, ollama_url: str = ""):
        self.model = model
        self.ollama_url = (ollama_url or OLLAMA_DEFAULT_URL).rstrip("/")

    def check(self) -> tuple[bool, str]:
        # 1) Ollama 서버 접근성 + 모델 확인
        ok, msg = _check_ollama(self.ollama_url, self.model)
        if not ok:
            return False, msg

        # 2) Anthropic 호환 엔드포인트(/v1/messages) 존재 여부 확인
        try:
            resp = requests.post(
                f"{self.ollama_url}/v1/messages",
                json={"model": self.model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
                timeout=10,
            )
            if resp.status_code in (200, 400):
                return True, f"Ollama Native 모드 준비됨 ({self.ollama_url}, 모델: {self.model})"
            return False, (
                f"Ollama Anthropic 호환 엔드포인트 미지원 (status={resp.status_code}). "
                f"Ollama v0.14 이상이 필요합니다. 'ollama --version'으로 확인하세요."
            )
        except requests.ConnectionError:
            return False, f"Ollama 서버({self.ollama_url})에 연결할 수 없습니다."
        except Exception as e:
            return False, f"Ollama Native 확인 중 오류: {e}"

    def activate(self, event_queue) -> None:
        os.environ["ANTHROPIC_BASE_URL"] = self.ollama_url
        os.environ["ANTHROPIC_API_KEY"] = "ollama"

        # count_tokens 이슈 우회: 모든 모델 슬롯을 동일 모델로 매핑
        os.environ["ANTHROPIC_DEFAULT_SONNET_MODEL"] = self.model
        os.environ["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = self.model
        os.environ["ANTHROPIC_DEFAULT_OPUS_MODEL"] = self.model

        event_queue.put(("status",
            f"[Native] Ollama Anthropic API 직접 연결 ({self.ollama_url}, 모델: {self.model})"))

    def cleanup(self, event_queue) -> None:
        for key in ("ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                     "ANTHROPIC_DEFAULT_OPUS_MODEL"):
            os.environ.pop(key, None)


# ──────────────────────────────────────────────
# 3) vLLM 서빙 엔진 경유 (프로덕션 고성능)
# ──────────────────────────────────────────────

class VllmStrategy(ProxyStrategy):
    """vLLM 서빙 엔진을 통한 고성능 전략.

    vLLM이 OpenAI 호환 API로 모델을 서빙하고,
    claude-code-proxy가 Claude API → OpenAI API 변환을 담당한다.

    장점:
    - Guided Decoding(Outlines)으로 tool call JSON 100% 유효성 보장
    - PagedAttention으로 높은 동시 처리량
    - 양자화(AWQ/GPTQ/FP8) 지원으로 GPU 메모리 효율화
    """

    def __init__(
        self,
        model: str,
        vllm_url: str = "",
        api_key: str = "",
    ):
        super().__init__(
            llm_provider="vLLM",
            model=model,
            api_key=api_key,
        )
        self.vllm_url = (vllm_url or VLLM_DEFAULT_URL).rstrip("/")

    # ── check ──

    def check(self) -> tuple[bool, str]:
        import shutil

        # 1) claude-code-proxy CLI 존재 확인
        if not shutil.which("claude-code-proxy"):
            return False, (
                "claude-code-proxy CLI를 찾을 수 없습니다. "
                "'pip install claude-code-proxy'로 설치하세요."
            )

        # 2) vLLM 서버 접근성 + 모델 확인
        try:
            resp = requests.get(f"{self.vllm_url}/v1/models", timeout=5)
            if resp.status_code != 200:
                return False, f"vLLM 서버 응답 오류 (status={resp.status_code})"
            models = [m["id"] for m in resp.json().get("data", [])]
            if self.model not in models:
                model_list = ", ".join(models[:10])
                return False, (
                    f"vLLM에 모델 '{self.model}'이 로드되지 않았습니다.\n"
                    f"사용 가능: {model_list}"
                )
        except requests.ConnectionError:
            return False, (
                f"vLLM 서버({self.vllm_url})에 연결할 수 없습니다. "
                f"'vllm serve {self.model}' 실행 여부를 확인하세요."
            )
        except Exception as e:
            return False, f"vLLM 연결 확인 중 오류: {e}"

        return True, f"vLLM 모드 준비됨 ({self.vllm_url}, 모델: {self.model})"

    # ── activate ──

    def activate(self, event_queue) -> None:
        import shutil

        event_queue.put(("status", f"[vLLM] claude-code-proxy → vLLM({self.vllm_url}) 연결 중..."))

        proxy_env = os.environ.copy()
        proxy_env["PORT"] = str(_PROXY_PORT)
        proxy_env["PREFERRED_PROVIDER"] = "openai"
        proxy_env["BIG_MODEL"] = self.model
        proxy_env["SMALL_MODEL"] = self.model
        proxy_env["OPENAI_API_KEY"] = self.api_key or "dummy"
        proxy_env["OPENAI_BASE_URL"] = f"{self.vllm_url}/v1"

        os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{_PROXY_PORT}"
        os.environ["ANTHROPIC_API_KEY"] = self.api_key or "dummy"

        log_tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", prefix="vllm_proxy_log_", delete=False
        )
        self._log_path = log_tmp.name
        log_tmp.close()
        self._log_file = open(self._log_path, "w", encoding="utf-8")

        proxy_path = shutil.which("claude-code-proxy") or "claude-code-proxy"
        self._proxy_process = subprocess.Popen(
            [proxy_path],
            env=proxy_env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )

        self._wait_for_health(event_queue)


# ──────────────────────────────────────────────
# 전략 선택 함수
# ──────────────────────────────────────────────

def select_strategy(
    llm_provider: str,
    backend_mode: str,
    model: str,
    api_key: str = "",
    ollama_url: str = "",
    vllm_url: str = "",
) -> BackendStrategy:
    """프로바이더와 백엔드 모드에 따라 적절한 전략을 선택하여 반환한다.

    Args:
        llm_provider: "Ollama" | "Gemini API" | "vLLM"
        backend_mode: "auto" | "proxy" | "native" (vLLM일 경우 무시됨)
        model: 사용할 모델명
        api_key: API 키 (Gemini)
        ollama_url: Ollama 서버 URL
        vllm_url: vLLM 서버 URL

    Returns:
        BackendStrategy 인스턴스
    """
    # vLLM → 전용 전략 (claude-code-proxy 경유)
    if llm_provider == "vLLM":
        return VllmStrategy(model, vllm_url, api_key)

    # native 모드 요청
    if backend_mode == "native":
        if llm_provider == "Ollama":
            return OllamaNativeStrategy(model, ollama_url)
        # Ollama 외 프로바이더는 native 미지원 → proxy 폴백
        return ProxyStrategy(llm_provider, model, api_key, ollama_url)

    # proxy 모드 요청
    if backend_mode == "proxy":
        return ProxyStrategy(llm_provider, model, api_key, ollama_url)

    # auto 모드: Ollama면 Native 시도 → 실패 시 Proxy 폴백
    if llm_provider == "Ollama":
        native = OllamaNativeStrategy(model, ollama_url)
        ok, _ = native.check()
        if ok:
            return native
        return ProxyStrategy(llm_provider, model, api_key, ollama_url)

    # Gemini 등 기타 → proxy
    return ProxyStrategy(llm_provider, model, api_key, ollama_url)


# ──────────────────────────────────────────────
# 공유 유틸리티 (모듈 레벨)
# ──────────────────────────────────────────────

def _check_ollama(ollama_url: str, model: str) -> tuple[bool, str]:
    """Ollama 서버 접근성 및 모델 존재 여부를 확인한다."""
    base = ollama_url.rstrip("/")
    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False, f"Ollama 서버 응답 오류 (status={resp.status_code})"
        models = [m["name"] for m in resp.json().get("models", [])]
        model_base = model.split(":")[0] if ":" in model else model
        found = any(model_base in m for m in models)
        if not found:
            return False, (
                f"Ollama 서버에 모델 '{model}'이 없습니다.\n"
                f"사용 가능: {', '.join(models[:10])}"
            )
        return True, f"Ollama 연결됨 (모델: {model})"
    except requests.ConnectionError:
        return False, f"Ollama 서버({base})에 연결할 수 없습니다. 'ollama serve' 실행 여부를 확인하세요."
    except Exception as e:
        logger.exception("Ollama 연결 확인 중 오류")
        return False, f"Ollama 연결 확인 중 오류: {e}"


def _read_log_tail(log_path: str, max_chars: int = 500) -> str:
    """로그 파일의 마지막 N자를 읽어 반환한다."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            return content[-max_chars:] if len(content) > max_chars else content
    except Exception:
        return "(로그 읽기 실패)"


def _detect_dropped_params(log_path: str) -> str:
    """프록시 로그에서 드롭된 파라미터를 감지하여 목록 문자열을 반환한다."""
    dropped_params: set[str] = set()
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                lower = line.lower()
                if "drop" in lower and "param" in lower:
                    bracket_match = re.search(r"\[([^\]]+)\]", line)
                    if bracket_match:
                        params = bracket_match.group(1)
                        for p in params.replace("'", "").replace('"', "").split(","):
                            p = p.strip()
                            if p:
                                dropped_params.add(p)
    except Exception:
        logger.debug("프록시 로그 파싱 중 오류", exc_info=True)
    return ", ".join(sorted(dropped_params))


def _safe_unlink(path: Optional[str]) -> None:
    """파일을 안전하게 삭제한다."""
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass

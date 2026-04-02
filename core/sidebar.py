"""
사이드바 - LLM 프로바이더 선택 UI
모든 앱에서 공유하는 LLM 설정 사이드바 컴포넌트
"""

import os
import streamlit as st
from llm_clients import OllamaClient, GeminiClient


def render_llm_sidebar() -> dict:
    """
    LLM 프로바이더 선택 UI를 렌더링하고 설정값 딕셔너리를 반환합니다.

    Returns:
        dict: {
            "llm_provider": str,
            "model_name": str,
            "api_key": str,
            "ollama_url": str,
            "vllm_url": str,
            "enable_thinking": bool,
            "agent_mode": str,
            "backend_mode": str,
        }
    """
    st.markdown("---")
    st.markdown("### ⚙️ LLM 프로바이더 설정")
    llm_provider = st.radio(
        "Provider 선택",
        ["Ollama", "Gemini API", "vLLM"],
        horizontal=True,
    )

    # 프로바이더별 초기값
    ollama_url = "http://localhost:11434"
    vllm_url = "http://localhost:8000"
    api_key = ""
    model_name = ""

    if llm_provider == "Ollama":
        ollama_url = st.text_input(
            "Ollama URL",
            value="http://localhost:11434",
            help="Ollama 서버 주소",
        )
        model_name = st.text_input(
            "모델",
            value="qwen3:8b",
            help="Ollama에 설치된 모델명 (에이전트 모드는 tool calling 지원 모델 필요: qwen3:8b, llama3.1 등)",
        )
        if st.button("🔌 연결 테스트", use_container_width=True):
            client = OllamaClient(ollama_url, model_name)
            ok, msg = client.check_connection()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    elif llm_provider == "Gemini API":
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Google AI Studio 등에서 발급받은 API 키",
        )
        model_name = st.selectbox(
            "모델",
            [
                "gemini-2.5-flash-lite",
            ],
        )
        if st.button("🔌 연결 테스트", use_container_width=True):
            client = GeminiClient(api_key, model_name)
            ok, msg = client.check_connection()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    else:  # vLLM
        vllm_url = st.text_input(
            "vLLM 서버 URL",
            value="http://localhost:8000",
            help="vLLM 서빙 엔진 주소 (OpenAI 호환 API)",
        )
        model_name = st.text_input(
            "모델명",
            value="Qwen/Qwen3-8B",
            help="vLLM에 로드된 모델 ID (HuggingFace 이름)",
        )
        st.info(
            "💡 vLLM 서버 시작 예시:\n"
            "```\n"
            "vllm serve Qwen/Qwen3-8B \\\n"
            "  --enable-auto-tool-choice \\\n"
            "  --tool-call-parser hermes \\\n"
            "  --guided-decoding-backend outlines\n"
            "```"
        )
        if st.button("🔌 연결 테스트", use_container_width=True):
            import requests as _req
            try:
                resp = _req.get(f"{vllm_url.rstrip('/')}/v1/models", timeout=5)
                if resp.status_code == 200:
                    models = [m["id"] for m in resp.json().get("data", [])]
                    if model_name in models:
                        st.success(f"vLLM 연결 성공 (모델: {model_name})")
                    else:
                        st.warning(f"서버 연결됨. 사용 가능 모델: {', '.join(models)}")
                else:
                    st.error(f"vLLM 응답 오류 (status={resp.status_code})")
            except _req.ConnectionError:
                st.error(f"vLLM 서버({vllm_url})에 연결할 수 없습니다.")
            except Exception as e:
                st.error(f"연결 오류: {e}")

    st.markdown("---")
    st.markdown("### 🎛️ 에이전트 설정")

    enable_thinking = st.toggle(
        "Thinking 모드",
        value=False,
        help="Qwen3의 /think 모드 활성화 (더 정확하지만 느림)",
    )

    # 백엔드 모드 (vLLM 이외 프로바이더에서만 표시)
    backend_mode = "auto"
    if llm_provider not in ("vLLM",):
        st.markdown("---")
        st.markdown("### 🔌 백엔드 모드")
        backend_mode = st.radio(
            "백엔드 선택",
            options=["auto", "proxy", "native"],
            index=0,
            help=(
                "auto: Native 시도 후 실패 시 Proxy 폴백\n"
                "proxy: claude-code-proxy 경유 (안정적)\n"
                "native: Ollama Anthropic API 직접 연결 (실험적, Ollama v0.14+)"
            ),
            horizontal=True,
        )

    st.markdown("---")
    st.markdown("### 🔧 실행 모드")

    agent_mode = st.radio(
        "모드 선택",
        options=["🤖 에이전트 (계획→실행→검증)", "💬 채팅 (간단한 도구 사용)"],
        index=0,
        help="에이전트: To-do 계획 수립 후 순차 실행\n채팅: 자유로운 대화 + 필요시 도구 사용",
    )

    return {
        "llm_provider": llm_provider,
        "model_name": model_name,
        "api_key": api_key,
        "ollama_url": ollama_url,
        "vllm_url": vllm_url,
        "enable_thinking": enable_thinking,
        "agent_mode": agent_mode,
        "backend_mode": backend_mode,
    }

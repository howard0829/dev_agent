"""
사이드바 - LLM 프로바이더 선택 UI
모든 앱에서 공유하는 LLM 설정 사이드바 컴포넌트
"""

import os
import streamlit as st
from llm_clients import OllamaClient, GeminiClient, OpenRouterClient
from agent import ClaudeAgentRunner


def render_llm_sidebar() -> dict:
    """
    LLM 프로바이더 선택 UI를 렌더링하고 설정값 딕셔너리를 반환합니다.

    Returns:
        dict: {
            "llm_provider": str,
            "model_name": str,
            "api_key": str,
            "ollama_url": str,
            "claude_model": str,
            "enable_thinking": bool,
            "agent_mode": str,
        }
    """
    st.markdown("---")
    st.markdown("### ⚙️ LLM 프로바이더 설정")
    llm_provider = st.radio(
        "Provider 선택",
        ["Ollama", "Gemini API", "OpenRouter", "Claude"],
        horizontal=True,
    )

    # 프로바이더별 초기값
    ollama_url = "http://localhost:11434"
    api_key = ""
    claude_model = "sonnet"
    model_name = ""

    if llm_provider == "Ollama":
        ollama_url = st.text_input(
            "Ollama URL",
            value="http://localhost:11434",
            help="Ollama 서버 주소",
        )
        model_name = st.text_input(
            "모델",
            value="qwen3-vl:2b",
            help="Ollama에 설치된 모델명 (예: qwen3-vl:2b)",
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

    elif llm_provider == "OpenRouter":
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=os.getenv("OPENROUTER_API_KEY", ""),
            help="openrouter.ai에서 발급받은 API Key",
        )
        model_name = st.selectbox(
            "모델",
            [
                # 코딩 특화
                "qwen/qwen3-coder:free",
                # 대형 범용 모델
                "nvidia/nemotron-3-super-120b-a12b:free",
                "qwen/qwen3-next-80b-a3b-instruct:free",
                "openai/gpt-oss-120b:free",
                "nousresearch/hermes-3-llama-3.1-405b:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                # 중형 범용 모델
                "stepfun/step-3.5-flash:free",
                "google/gemma-3-27b-it:free",
                "minimax/minimax-m2.5:free",
                # 경량/빠른 응답
                "nvidia/nemotron-nano-9b-v2:free",
                "google/gemma-3-12b-it:free",
                "google/gemma-3-4b-it:free",
            ],
            help="OpenRouter 무료 티어 모델 (일 50회 제한)",
        )
        if st.button("🔌 연결 테스트", use_container_width=True):
            client = OpenRouterClient(api_key, model_name)
            ok, msg = client.check_connection()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    else:  # Claude
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            help="Anthropic에서 발급받은 API Key",
        )
        claude_model = st.selectbox(
            "Claude 모델",
            ["sonnet", "opus", "haiku"],
            help="Claude Agent SDK 모델 (sonnet 권장)",
        )
        model_name = f"claude-{claude_model}"
        if st.button("🔌 연결 테스트", use_container_width=True):
            runner = ClaudeAgentRunner(
                llm_provider="Claude", api_key=api_key, model=claude_model
            )
            ok, msg = runner.check_connection()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.markdown("---")
    st.markdown("### 🎛️ 에이전트 설정")

    enable_thinking = st.toggle(
        "Thinking 모드",
        value=False,
        help="Qwen3의 /think 모드 활성화 (더 정확하지만 느림)",
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
        "claude_model": claude_model,
        "enable_thinking": enable_thinking,
        "agent_mode": agent_mode,
    }

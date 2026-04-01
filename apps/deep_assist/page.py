"""
DeepAssist 페이지 모듈
자율 코딩 에이전트 — 사이드바(LLM 설정) + 메인(채팅/워크스페이스 탭)
core/ 유틸리티를 재사용합니다.
"""

import streamlit as st

from core.session import init_session, reset_chat, reset_logs, display_path, get_state
from core.sidebar import render_llm_sidebar
from core.chat_ui import render_chat_tab
from core.workspace_ui import render_workspace_tab


# ──────────────────────────────────────────────
# 세션 초기화
# ──────────────────────────────────────────────

def init_app_session(prefix: str):
    """DeepAssist 세션 상태 초기화"""
    init_session(prefix)


# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────

def render_sidebar(prefix: str) -> dict:
    """
    DeepAssist 사이드바 렌더링.
    LLM 프로바이더 선택 + 에이전트 설정 + 제어 버튼.

    Returns:
        dict: provider_cfg (llm_provider, model_name, api_key, ...)
    """
    st.markdown("## 🤖 DeepAssist")
    st.caption("Ollama · Gemini · OpenRouter · Claude")

    # 공통 LLM 프로바이더 선택 UI 재사용
    provider_cfg = render_llm_sidebar()

    # 제어 버튼
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            reset_chat(prefix)
            st.rerun()
    with col2:
        if st.button("📋 로그 지우기", use_container_width=True):
            reset_logs(prefix)
            st.rerun()

    return provider_cfg


# ──────────────────────────────────────────────
# 메인 화면
# ──────────────────────────────────────────────

def render_main(prefix: str, sidebar_cfg: dict):
    """DeepAssist 메인 화면 렌더링: 채팅 + 워크스페이스 탭"""
    model_name = sidebar_cfg["model_name"]
    working_dir = get_state(prefix, "working_dir")
    st.caption(f"모델: `{model_name}` | 작업 디렉토리: `{display_path(working_dir)}`")

    tab_chat, tab_ws = st.tabs(["💬 채팅", "📁 워크스페이스"])

    with tab_chat:
        render_chat_tab(prefix, sidebar_cfg)

    with tab_ws:
        render_workspace_tab(prefix)

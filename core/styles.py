"""
공통 CSS 스타일
모든 앱에서 공유하는 기본 스타일 정의
"""

import streamlit as st


COMMON_CSS = """
<style>
    /* 전체 레이아웃 */
    .main .block-container { max-width: 1100px; padding-top: 1rem; }

    /* ── 앱 스위처 카드 버튼 ── */
    div[data-testid="stColumns"]:has(button[data-testid="stBaseButton-secondary"]) {
        gap: 12px !important;
    }

    /* 공통 카드 버튼 스타일 */
    .app-card-btn button {
        border-radius: 14px !important;
        padding: 16px 12px !important;
        min-height: 80px !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        transition: all 0.25s cubic-bezier(.4,0,.2,1) !important;
        backdrop-filter: blur(12px);
        line-height: 1.4 !important;
        white-space: pre-wrap !important;
    }

    /* 비활성 카드 */
    .app-card-btn button[kind="secondary"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1.5px solid rgba(255,255,255,0.08) !important;
        color: #8b949e !important;
    }
    .app-card-btn button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.07) !important;
        border-color: rgba(88,166,255,0.35) !important;
        color: #c9d1d9 !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }

    /* 활성 카드 */
    .app-card-btn button[kind="primary"] {
        background: linear-gradient(135deg, rgba(88,166,255,0.15) 0%, rgba(63,185,80,0.08) 100%) !important;
        border: 1.5px solid #58a6ff !important;
        color: #e6edf3 !important;
        box-shadow: 0 0 24px rgba(88,166,255,0.12), 0 4px 12px rgba(0,0,0,0.1) !important;
        cursor: default !important;
    }

    /* 카드 아래 설명 텍스트 */
    .app-card-desc {
        text-align: center;
        font-size: 0.72rem;
        color: #8b949e;
        margin-top: -8px;
        margin-bottom: 8px;
        line-height: 1.3;
    }

    /* 상태 메시지 */
    .status-msg {
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.82rem;
        color: #8b949e;
        padding: 2px 0;
        line-height: 1.5;
    }
    .status-msg.error { color: #f85149; }

    /* 도구 호출 카드 */
    .tool-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.8rem;
    }
    .tool-name {
        color: #58a6ff;
        font-weight: 600;
    }
    .tool-args { color: #8b949e; }
    .tool-result {
        color: #c9d1d9;
        white-space: pre-wrap;
        word-break: break-all;
        max-height: 200px;
        overflow-y: auto;
    }

    /* 플랜 체크리스트 */
    .plan-item {
        padding: 4px 8px;
        margin: 2px 0;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    .plan-pending { background: #1c1c1c; color: #8b949e; }
    .plan-running { background: #0d1117; border-left: 3px solid #58a6ff; color: #58a6ff; }
    .plan-done { background: #0d1117; color: #3fb950; }
    .plan-failed { background: #0d1117; color: #f85149; }

    /* 채팅 메시지 */
    .stChatMessage { max-width: 100% !important; }

    /* 사이드바 - 밝은 톤 */
    section[data-testid="stSidebar"] {
        background: #f4f6f9;
        min-width: 320px;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #1a56db;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
</style>
"""


def apply_common_styles():
    """공통 CSS 스타일 적용"""
    st.markdown(COMMON_CSS, unsafe_allow_html=True)


def apply_custom_css(css: str):
    """앱별 커스텀 CSS 추가 적용"""
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

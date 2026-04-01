"""
세션 상태 네임스페이스 헬퍼
앱별 독립 세션 상태를 관리하여 앱 전환 시 데이터 충돌을 방지합니다.
"""

import os
import streamlit as st


def ns(prefix: str, key: str) -> str:
    """네임스페이스 키 생성"""
    return f"{prefix}.{key}"


def get_state(prefix: str, key: str, default=None):
    """네임스페이스된 세션 상태 값 조회"""
    return st.session_state.get(ns(prefix, key), default)


def set_state(prefix: str, key: str, value):
    """네임스페이스된 세션 상태 값 설정"""
    st.session_state[ns(prefix, key)] = value


def init_session(prefix: str):
    """앱별 세션 상태 초기화 (이미 존재하면 스킵)"""
    defaults = {
        "messages": [],
        "agent": None,
        "client": None,
        "status_log": [],
        "tool_log": [],
        "todo_items": [],
        "current_plan": None,
        "is_running": False,
        "working_dir": os.path.expanduser("~"),
        "editing_file": None,
        "_uploaded_hash": None,
    }
    for k, v in defaults.items():
        key = ns(prefix, k)
        if key not in st.session_state:
            st.session_state[key] = v


def reset_chat(prefix: str):
    """채팅 상태 초기화"""
    set_state(prefix, "messages", [])
    set_state(prefix, "status_log", [])
    set_state(prefix, "tool_log", [])
    set_state(prefix, "todo_items", [])
    set_state(prefix, "current_plan", None)
    agent = get_state(prefix, "agent")
    if agent and hasattr(agent, "reset_history"):
        agent.reset_history()


def reset_logs(prefix: str):
    """로그만 초기화"""
    set_state(prefix, "status_log", [])
    set_state(prefix, "tool_log", [])


def display_path(full_path: str) -> str:
    """절대 경로에서 /workspaces/ 이후만 추출하여 UI에 표시"""
    idx = full_path.find("/workspaces/")
    if idx != -1:
        return full_path[idx:]
    return full_path

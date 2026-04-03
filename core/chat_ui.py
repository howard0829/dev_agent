"""
채팅 탭 - 채팅 UI, 콜백, 에이전트 실행 로직
모든 앱에서 공유하는 채팅 인터페이스 컴포넌트
"""

import os
import datetime
import streamlit as st

from models import Plan, ToolCallRecord, ProviderConfig, CallbackSet
from llm_clients import OllamaClient, GeminiClient
from agent import ClaudeAgentRunner
from core.session import ns, get_state, set_state
from config import VLLM_DEFAULT_URL, AGENT_MAX_TURNS


# ──────────────────────────────────────────────
# 콜백 팩토리
# ──────────────────────────────────────────────

def _make_callbacks(prefix: str, todo_placeholder, status_log_placeholder) -> CallbackSet:
    """앱별 네임스페이스 콜백 함수 세트를 생성하여 반환"""

    def _update_todo_ui():
        with todo_placeholder.container():
            items = get_state(prefix, "todo_items", [])
            if items:
                total = len(items)
                done = sum(1 for t in items if t["status"] == "completed")
                st.markdown(f"### 📋 Todo ({done}/{total})")
                st.progress(done / total if total > 0 else 0)
                for t in items:
                    status = t.get("status", "pending")
                    if status == "completed":
                        icon = "✅"
                    elif status == "in_progress":
                        icon = "🔄"
                    else:
                        icon = "⬜"
                    st.markdown(f"{icon} {t['text']}")

    def _update_status_log_ui():
        with status_log_placeholder.container():
            st.markdown("### 📊 상태 로그")
            log = get_state(prefix, "status_log", [])
            if log:
                log_text = "\n".join(log)
                with st.container(height=500):
                    st.code(log_text, language="text")
            else:
                st.caption("실행 로그가 없습니다.")

    def on_status(msg: str):
        log = get_state(prefix, "status_log", [])
        log.append(msg)
        set_state(prefix, "status_log", log)
        _update_status_log_ui()

    def on_todo_update(items: list):
        set_state(prefix, "todo_items", items)
        _update_todo_ui()

    def on_tool_call(record: ToolCallRecord):
        tool_log = get_state(prefix, "tool_log", [])
        tool_log.append(record.to_dict())
        set_state(prefix, "tool_log", tool_log)

    def on_plan_update(plan: Plan):
        set_state(prefix, "current_plan", plan.to_dict())

    return {
        "on_status": on_status,
        "on_todo_update": on_todo_update,
        "on_tool_call": on_tool_call,
        "on_plan_update": on_plan_update,
        "update_todo_ui": _update_todo_ui,
        "update_status_log_ui": _update_status_log_ui,
    }


# ──────────────────────────────────────────────
# 에이전트 생성
# ──────────────────────────────────────────────

def _get_agent(prefix: str, provider_cfg: ProviderConfig, callbacks: CallbackSet, is_agent_mode: bool):
    """프로바이더 설정에 따라 에이전트/클라이언트 인스턴스를 생성하여 반환"""
    llm_provider = provider_cfg["llm_provider"]
    model_name = provider_cfg["model_name"]
    api_key = provider_cfg["api_key"]
    ollama_url = provider_cfg["ollama_url"]
    vllm_url = provider_cfg.get("vllm_url", VLLM_DEFAULT_URL)
    backend_mode = provider_cfg.get("backend_mode", "auto")
    working_dir = get_state(prefix, "working_dir", os.path.expanduser("~"))

    if is_agent_mode:
        runner = ClaudeAgentRunner(
            llm_provider=llm_provider,
            api_key=api_key,
            model=model_name,
            ollama_url=ollama_url if llm_provider == "Ollama" else None,
            vllm_url=vllm_url if llm_provider == "vLLM" else None,
            max_turns=AGENT_MAX_TURNS,
            working_dir=working_dir,
            on_status=callbacks["on_status"],
            on_tool_call=callbacks["on_tool_call"],
            on_plan_update=callbacks["on_plan_update"],
            backend_mode=backend_mode,
        )
        runner.on_todo_update = callbacks["on_todo_update"]
        return runner
    else:
        # 단순 챗
        if llm_provider == "Ollama":
            return OllamaClient(ollama_url, model_name)
        elif llm_provider == "Gemini API":
            return GeminiClient(api_key, model_name)
        else:
            # vLLM 등 — 에이전트 러너를 채팅 모드로 사용
            return ClaudeAgentRunner(
                llm_provider=llm_provider,
                api_key=api_key,
                model=model_name,
                vllm_url=vllm_url if llm_provider == "vLLM" else None,
                working_dir=working_dir,
                max_turns=AGENT_MAX_TURNS,
            )


# ──────────────────────────────────────────────
# 채팅 탭 렌더링
# ──────────────────────────────────────────────

def render_chat_tab(prefix: str, provider_cfg: ProviderConfig):
    """채팅 탭 전체를 렌더링합니다."""

    # 2컬럼 레이아웃: 채팅 + 상태 패널
    chat_col, status_col = st.columns([3, 1])

    with status_col:
        todo_placeholder = st.empty()
        status_log_placeholder = st.empty()

    # 콜백 생성
    callbacks = _make_callbacks(prefix, todo_placeholder, status_log_placeholder)

    # 초기 상태 렌더링
    callbacks["update_todo_ui"]()
    callbacks["update_status_log_ui"]()

    # ── 채팅 UI ──
    with chat_col:
        messages = get_state(prefix, "messages", [])
        # 메시지 표시
        for msg in messages:
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])
                if role == "assistant" and msg.get("tool_calls"):
                    with st.expander("🔧 도구 호출 기록", expanded=False):
                        for tc in msg["tool_calls"]:
                            st.markdown(f"**`{tc['name']}`**")
                            st.code(str(tc.get("args", {}))[:200], language="json")
                            if tc.get("result"):
                                st.code(tc["result"][:200], language="text")
                            st.markdown("---")

        # 입력창 (실행 중이 아닐 때만 활성화)
        is_running = get_state(prefix, "is_running", False)
        if not is_running:
            prompt = st.chat_input("무엇을 도와드릴까요?", key=f"{prefix}_chat_input")
        else:
            st.chat_input("에이전트 실행 중...", disabled=True, key=f"{prefix}_chat_input_disabled")
            prompt = None

        if prompt:
            _handle_prompt(prefix, prompt, provider_cfg, callbacks)


def _handle_prompt(prefix: str, prompt: str, provider_cfg: dict, callbacks: dict):
    """사용자 입력 처리: 에이전트/채팅 모드에 따라 실행"""
    llm_provider = provider_cfg["llm_provider"]
    enable_thinking = provider_cfg["enable_thinking"]
    agent_mode = provider_cfg["agent_mode"]

    set_state(prefix, "is_running", True)
    set_state(prefix, "status_log", [])
    set_state(prefix, "tool_log", [])
    set_state(prefix, "todo_items", [])
    set_state(prefix, "current_plan", None)

    # 메시지 추가
    messages = get_state(prefix, "messages", [])
    messages.append({"role": "user", "content": prompt})
    set_state(prefix, "messages", messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    is_agent_mode = agent_mode.startswith("🤖")
    agent = _get_agent(prefix, provider_cfg, callbacks, is_agent_mode)

    main_log_container = st.empty()
    live_logs: list[str] = []

    # 콜백 래핑은 ClaudeAgentRunner(에이전트 모드)에서만 적용
    if hasattr(agent, "on_tool_call"):
        orig_on_tool_call = agent.on_tool_call
        def live_on_tool_call(record: ToolCallRecord):
            orig_on_tool_call(record)
            live_logs.append(f"🔧 **{record.tool_name}** 호출 완료")
            with main_log_container.container():
                st.info("\n\n".join(live_logs[-8:]))
        agent.on_tool_call = live_on_tool_call

    if hasattr(agent, "on_status"):
        orig_on_status = agent.on_status
        def live_on_status(msg: str):
            orig_on_status(msg)
            live_logs.append(msg)
            with main_log_container.container():
                st.info("\n\n".join(live_logs[-8:]))
        agent.on_status = live_on_status

    if is_agent_mode:
        # ── 에이전트 모드: 연결 확인 + 스피너 + SDK 호출 ──
        ok, conn_msg = agent.check_connection()
        if not ok:
            response = f"❌ 에이전트 연결 실패:\n\n{conn_msg}"
            with st.chat_message("assistant"):
                st.markdown(response)
            messages = get_state(prefix, "messages", [])
            messages.append({"role": "assistant", "content": response, "tool_calls": [], "plan": None})
            set_state(prefix, "messages", messages)
            set_state(prefix, "is_running", False)
            st.rerun()
            return

        with st.spinner("🤖 에이전트 작업 중..."):
            try:
                response = agent.run(prompt)
            except Exception as e:
                response = f"❌ 실행 중 오류가 발생했습니다:\n```\n{e}\n```"

        main_log_container.empty()

        # 상태 로그 파일 저장
        status_log = get_state(prefix, "status_log", [])
        if status_log:
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"status_log_{now_str}.txt"
            working_dir = get_state(prefix, "working_dir", os.path.expanduser("~"))
            log_path = os.path.join(working_dir, log_filename)
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(status_log))
                status_log.append(f"📁 상태 로그 저장 완료: {log_filename}")
                set_state(prefix, "status_log", status_log)
            except Exception as e:
                status_log.append(f"❌ 상태 로그 저장 실패: {e}")
                set_state(prefix, "status_log", status_log)

        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        # ── 단순 채팅 모드: 스트리밍 출력 ──
        with st.chat_message("assistant"):
            if hasattr(agent, "run"):
                # ClaudeAgentRunner (채팅 모드)
                with st.spinner("💬 응답 생성 중..."):
                    try:
                        response = agent.chat(prompt)
                    except Exception as e:
                        response = f"❌ 오류:\n```\n{e}\n```"
                st.markdown(response)
            else:
                # OllamaClient / GeminiClient — UI 스트리밍
                msgs = [
                    {"role": m["role"], "content": str(m.get("content", ""))}
                    for m in get_state(prefix, "messages", [])
                ]
                try:
                    response = st.write_stream(
                        agent.stream_chat(
                            messages=msgs,
                            enable_thinking=enable_thinking,
                        )
                    )
                except Exception as e:
                    response = f"❌ 오류:\n```\n{e}\n```"
                    st.markdown(response)

    # 메시지 저장 (도구 콜 포함)
    tool_log = get_state(prefix, "tool_log", [])
    msg_data = {
        "role": "assistant",
        "content": response,
        "tool_calls": [
            {"name": t["name"], "args": t["args"], "result": t["result"][:200]}
            for t in tool_log
        ],
        "plan": get_state(prefix, "current_plan"),
    }
    messages = get_state(prefix, "messages", [])
    messages.append(msg_data)
    set_state(prefix, "messages", messages)

    set_state(prefix, "is_running", False)
    st.rerun()

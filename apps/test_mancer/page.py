"""
TestMancer 페이지 모듈
테스트 자동화 에이전트 — 자체 사이드바(테스트 설정) + 메인(채팅/테스트 결과 탭)
DeepAssist와 독립적인 UI/백엔드 구조를 가집니다.
"""

import os
import streamlit as st

from core.session import ns, get_state, set_state, display_path
from models import Plan, ToolCallRecord
from llm_clients import OllamaClient, GeminiClient, OpenRouterClient
from agent import ClaudeAgentRunner


# ──────────────────────────────────────────────
# 세션 초기화
# ──────────────────────────────────────────────

def init_app_session(prefix: str):
    """TestMancer 전용 세션 상태 초기화"""
    defaults = {
        # 공통
        "messages": [],
        "agent": None,
        "status_log": [],
        "tool_log": [],
        "todo_items": [],
        "current_plan": None,
        "is_running": False,
        "working_dir": os.path.expanduser("~"),
        # TestMancer 전용
        "test_results": [],         # 테스트 실행 결과 히스토리
        "test_framework": "pytest", # 선택된 테스트 프레임워크
        "test_target_dir": "",      # 테스트 대상 디렉토리
    }
    for k, v in defaults.items():
        key = ns(prefix, k)
        if key not in st.session_state:
            st.session_state[key] = v


# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────

def render_sidebar(prefix: str) -> dict:
    """
    TestMancer 전용 사이드바 렌더링.
    LLM 설정 + 테스트 프레임워크/대상 설정.

    Returns:
        dict: sidebar_cfg (llm 설정 + 테스트 설정)
    """
    st.markdown("## 🧪 TestMancer")
    st.caption("테스트 자동화 에이전트")

    # ── LLM 프로바이더 설정 ──
    st.markdown("---")
    st.markdown("### ⚙️ LLM 설정")
    llm_provider = st.radio(
        "Provider",
        ["Claude", "OpenRouter", "Ollama"],
        horizontal=True,
        key=f"{prefix}_llm_provider",
    )

    api_key = ""
    ollama_url = "http://localhost:11434"
    claude_model = "sonnet"
    model_name = ""

    if llm_provider == "Claude":
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            key=f"{prefix}_api_key",
        )
        claude_model = st.selectbox(
            "모델", ["sonnet", "opus", "haiku"],
            key=f"{prefix}_claude_model",
        )
        model_name = f"claude-{claude_model}"
    elif llm_provider == "OpenRouter":
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=os.getenv("OPENROUTER_API_KEY", ""),
            key=f"{prefix}_api_key",
        )
        model_name = st.selectbox(
            "모델",
            [
                "qwen/qwen3-coder:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemma-3-27b-it:free",
            ],
            key=f"{prefix}_model",
        )
    else:  # Ollama
        ollama_url = st.text_input(
            "Ollama URL", value="http://localhost:11434",
            key=f"{prefix}_ollama_url",
        )
        model_name = st.text_input(
            "모델", value="qwen3-vl:2b",
            key=f"{prefix}_model",
        )

    # ── 테스트 설정 (TestMancer 전용) ──
    st.markdown("---")
    st.markdown("### 🎯 테스트 설정")

    test_framework = st.selectbox(
        "테스트 프레임워크",
        ["pytest", "unittest", "jest", "mocha", "go test", "cargo test"],
        key=f"{prefix}_framework",
    )
    set_state(prefix, "test_framework", test_framework)

    test_target = st.text_input(
        "테스트 대상 디렉토리",
        value=get_state(prefix, "test_target_dir", ""),
        placeholder="예: ./tests 또는 ./src",
        key=f"{prefix}_test_target",
    )
    set_state(prefix, "test_target_dir", test_target)

    test_mode = st.radio(
        "실행 모드",
        ["🧪 테스트 생성+실행", "📝 테스트 생성만", "▶️ 기존 테스트 실행"],
        key=f"{prefix}_test_mode",
    )

    # ── 제어 버튼 ──
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 대화 초기화", use_container_width=True, key=f"{prefix}_reset"):
            set_state(prefix, "messages", [])
            set_state(prefix, "status_log", [])
            set_state(prefix, "tool_log", [])
            set_state(prefix, "todo_items", [])
            set_state(prefix, "current_plan", None)
            st.rerun()
    with col2:
        if st.button("📋 결과 지우기", use_container_width=True, key=f"{prefix}_clear"):
            set_state(prefix, "test_results", [])
            set_state(prefix, "status_log", [])
            st.rerun()

    return {
        "llm_provider": llm_provider,
        "model_name": model_name,
        "api_key": api_key,
        "ollama_url": ollama_url,
        "claude_model": claude_model,
        "enable_thinking": False,
        "agent_mode": "🤖 에이전트 (계획→실행→검증)",
        # TestMancer 전용
        "test_framework": test_framework,
        "test_target": test_target,
        "test_mode": test_mode,
    }


# ──────────────────────────────────────────────
# 메인 화면
# ──────────────────────────────────────────────

def render_main(prefix: str, sidebar_cfg: dict):
    """TestMancer 메인 화면: 채팅 + 테스트 결과 탭"""
    model_name = sidebar_cfg["model_name"]
    framework = sidebar_cfg["test_framework"]
    working_dir = get_state(prefix, "working_dir")
    st.caption(
        f"모델: `{model_name}` | 프레임워크: `{framework}` | "
        f"작업 디렉토리: `{display_path(working_dir)}`"
    )

    tab_chat, tab_results = st.tabs(["💬 채팅", "📊 테스트 결과"])

    with tab_chat:
        _render_chat(prefix, sidebar_cfg)

    with tab_results:
        _render_test_results(prefix)


# ──────────────────────────────────────────────
# 채팅 탭 (TestMancer 전용)
# ──────────────────────────────────────────────

def _render_chat(prefix: str, sidebar_cfg: dict):
    """TestMancer 채팅 탭 — 테스트 특화 프롬프트 안내 포함"""
    chat_col, status_col = st.columns([3, 1])

    with status_col:
        todo_placeholder = st.empty()
        status_log_placeholder = st.empty()

    callbacks = _make_callbacks(prefix, todo_placeholder, status_log_placeholder)
    callbacks["update_todo_ui"]()
    callbacks["update_status_log_ui"]()

    with chat_col:
        messages = get_state(prefix, "messages", [])
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("tool_calls"):
                    with st.expander("🔧 도구 호출 기록", expanded=False):
                        for tc in msg["tool_calls"]:
                            st.markdown(f"**`{tc['name']}`**")
                            if tc.get("result"):
                                st.code(tc["result"][:300], language="text")
                            st.markdown("---")

        is_running = get_state(prefix, "is_running", False)
        if not is_running:
            # 테스트 특화 플레이스홀더
            prompt = st.chat_input(
                "테스트할 내용을 입력하세요 (예: '이 모듈의 유닛 테스트를 작성해줘')",
                key=f"{prefix}_chat_input",
            )
        else:
            st.chat_input("에이전트 실행 중...", disabled=True, key=f"{prefix}_chat_disabled")
            prompt = None

        if prompt:
            _handle_prompt(prefix, prompt, sidebar_cfg, callbacks)


def _handle_prompt(prefix: str, prompt: str, sidebar_cfg: dict, callbacks: dict):
    """사용자 입력 처리 — 테스트 컨텍스트를 시스템 프롬프트에 주입"""
    import datetime

    set_state(prefix, "is_running", True)
    set_state(prefix, "status_log", [])
    set_state(prefix, "tool_log", [])
    set_state(prefix, "todo_items", [])

    messages = get_state(prefix, "messages", [])
    messages.append({"role": "user", "content": prompt})
    set_state(prefix, "messages", messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    agent = _create_agent(prefix, sidebar_cfg, callbacks)

    # 테스트 컨텍스트를 프롬프트에 주입
    framework = sidebar_cfg.get("test_framework", "pytest")
    test_target = sidebar_cfg.get("test_target", "")
    test_mode = sidebar_cfg.get("test_mode", "")
    context_prefix = (
        f"[테스트 프레임워크: {framework}]"
        f"{f' [대상: {test_target}]' if test_target else ''}"
        f" [모드: {test_mode}]\n\n"
    )
    enriched_prompt = context_prefix + prompt

    with st.spinner("🧪 TestMancer 작업 중..."):
        try:
            response = agent.run(enriched_prompt)
        except Exception as e:
            response = f"❌ 실행 중 오류:\n```\n{e}\n```"

    # 테스트 결과 히스토리에 추가
    test_results = get_state(prefix, "test_results", [])
    test_results.append({
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "prompt": prompt[:100],
        "response_preview": response[:200] if isinstance(response, str) else str(response)[:200],
        "tool_count": len(get_state(prefix, "tool_log", [])),
    })
    set_state(prefix, "test_results", test_results)

    with st.chat_message("assistant"):
        st.markdown(response)

    tool_log = get_state(prefix, "tool_log", [])
    msg_data = {
        "role": "assistant",
        "content": response,
        "tool_calls": [
            {"name": t["name"], "args": t["args"], "result": t["result"][:200]}
            for t in tool_log
        ],
    }
    messages = get_state(prefix, "messages", [])
    messages.append(msg_data)
    set_state(prefix, "messages", messages)

    set_state(prefix, "is_running", False)
    st.rerun()


# ──────────────────────────────────────────────
# 테스트 결과 탭 (TestMancer 전용)
# ──────────────────────────────────────────────

def _render_test_results(prefix: str):
    """테스트 실행 결과 히스토리 표시"""
    results = get_state(prefix, "test_results", [])

    if not results:
        st.info("🧪 아직 테스트 실행 기록이 없습니다. 채팅 탭에서 테스트를 요청해보세요.")
        return

    st.markdown(f"### 📊 테스트 실행 히스토리 ({len(results)}건)")

    for i, r in enumerate(reversed(results)):
        with st.expander(
            f"#{len(results)-i} — {r['time']} | 🔧 도구 {r['tool_count']}회",
            expanded=(i == 0),
        ):
            st.markdown(f"**요청:** {r['prompt']}")
            st.markdown("**결과 미리보기:**")
            st.code(r["response_preview"], language="text")


# ──────────────────────────────────────────────
# 에이전트 생성
# ──────────────────────────────────────────────

def _create_agent(prefix: str, sidebar_cfg: dict, callbacks: dict):
    """TestMancer 에이전트 인스턴스 생성"""
    llm_provider = sidebar_cfg["llm_provider"]
    api_key = sidebar_cfg["api_key"]
    model_name = sidebar_cfg["model_name"]
    claude_model = sidebar_cfg["claude_model"]
    ollama_url = sidebar_cfg["ollama_url"]
    working_dir = get_state(prefix, "working_dir", os.path.expanduser("~"))

    model_val = claude_model if llm_provider == "Claude" else model_name
    runner = ClaudeAgentRunner(
        llm_provider=llm_provider,
        api_key=api_key,
        model=model_val,
        ollama_url=ollama_url if llm_provider == "Ollama" else None,
        max_turns=150,
        working_dir=working_dir,
        on_status=callbacks["on_status"],
        on_tool_call=callbacks["on_tool_call"],
        on_plan_update=callbacks["on_plan_update"],
    )
    runner.on_todo_update = callbacks["on_todo_update"]
    return runner


# ──────────────────────────────────────────────
# 콜백 팩토리
# ──────────────────────────────────────────────

def _make_callbacks(prefix: str, todo_placeholder, status_log_placeholder):
    """TestMancer 전용 콜백 세트 생성"""

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
                    icon = {"completed": "✅", "in_progress": "🔄"}.get(status, "⬜")
                    st.markdown(f"{icon} {t['text']}")

    def _update_status_log_ui():
        with status_log_placeholder.container():
            st.markdown("### 📊 실행 로그")
            log = get_state(prefix, "status_log", [])
            if log:
                with st.container(height=500):
                    st.code("\n".join(log), language="text")
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
        tool_log.append({
            "name": record.tool_name,
            "args": record.arguments,
            "result": record.result,
            "time": record.timestamp,
        })
        set_state(prefix, "tool_log", tool_log)

    def on_plan_update(plan: Plan):
        set_state(prefix, "current_plan", {
            "goal": plan.goal,
            "tasks": [
                {"id": t.id, "desc": t.description, "status": t.status, "result": t.result}
                for t in plan.tasks
            ],
        })

    return {
        "on_status": on_status,
        "on_todo_update": on_todo_update,
        "on_tool_call": on_tool_call,
        "on_plan_update": on_plan_update,
        "update_todo_ui": _update_todo_ui,
        "update_status_log_ui": _update_status_log_ui,
    }

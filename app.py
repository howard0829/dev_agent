"""
DeepAssist - Streamlit UI
Ollama, Gemini API, OpenRouter, Claude Agent SDK 기반 코딩 에이전트
"""

import os
import streamlit as st
import httpx
from models import Plan, ToolCallRecord
from llm_clients import OllamaClient, GeminiClient, OpenRouterClient
from agent import ClaudeAgentRunner

# FastAPI 파일 서버 주소
FILE_SERVER_URL = os.getenv("FILE_SERVER_URL", "http://localhost:8000")

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="DeepAssist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 커스텀 CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* 전체 레이아웃 */
    .main .block-container { max-width: 1100px; padding-top: 1.5rem; }

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

    /* 사이드바 - 밝은 톤으로 변경 */
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
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State 초기화
# ──────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],          # 채팅 메시지 [{role, content, tool_calls?, plan?}]
        "agent": None,
        "client": None,
        "status_log": [],        # 실시간 상태 로그
        "tool_log": [],          # 도구 호출 기록
        "todo_items": [],        # Todo 체크리스트 [{text, done}]
        "current_plan": None,
        "is_running": False,
        "working_dir": os.path.expanduser("~"),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


def display_path(full_path: str) -> str:
    """절대 경로에서 /workspaces/ 이후만 추출하여 UI에 표시"""
    idx = full_path.find("/workspaces/")
    if idx != -1:
        return full_path[idx:]
    return full_path


# ──────────────────────────────────────────────
# 사이드바 - 설정
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 DeepAssist")
    st.caption("Ollama · Gemini · OpenRouter · Claude")

    st.markdown("---")
    st.markdown("### ⚙️ LLM 프로바이더 설정")
    llm_provider = st.radio(
        "Provider 선택",
        ["Ollama", "Gemini API", "OpenRouter", "Claude"],
        horizontal=True,
    )

    # 프로바이더별 초기값
    ollama_url = "http://localhost:11434"
    gemini_api_key = ""
    openrouter_api_key = ""
    claude_api_key = ""
    claude_model = "sonnet"
    model_name = ""

    if llm_provider == "Ollama":
        ollama_url = st.text_input(
            "Ollama URL",
            value="http://localhost:11434",
            help="Ollama 서버 주소"
        )
        model_name = st.text_input(
            "모델",
            value="qwen3-vl:2b",
            help="Ollama에 설치된 모델명 (예: qwen3-vl:2b)"
        )
        if st.button("🔌 연결 테스트", use_container_width=True):
            client = OllamaClient(ollama_url, model_name)
            ok, msg = client.check_connection()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    elif llm_provider == "Gemini API":
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Google AI Studio 등에서 발급받은 API 키"
        )
        model_name = st.selectbox(
            "모델",
            [
                # "gemini-2.5-pro",
                # "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
            ]
        )
        if st.button("🔌 연결 테스트", use_container_width=True):
            client = GeminiClient(gemini_api_key, model_name)
            ok, msg = client.check_connection()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    elif llm_provider == "OpenRouter":
        openrouter_api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=os.getenv("OPENROUTER_API_KEY", ""),
            help="openrouter.ai에서 발급받은 API Key"
        )
        model_name = st.selectbox(
                "모델",
                [
                    # 코딩 특화
                    "qwen/qwen3-coder:free",                         # Qwen3 Coder 480B (ctx 262K)

                    # 대형 범용 모델
                    "nvidia/nemotron-3-super-120b-a12b:free",        # Nemotron 3 Super 120B (ctx 262K)
                    "qwen/qwen3-next-80b-a3b-instruct:free",        # Qwen3 Next 80B (ctx 262K)
                    "openai/gpt-oss-120b:free",                      # GPT-OSS 120B (ctx 131K)
                    "nousresearch/hermes-3-llama-3.1-405b:free",     # Hermes 3 405B (ctx 131K)
                    "meta-llama/llama-3.3-70b-instruct:free",        # Llama 3.3 70B (ctx 65K)

                    # 중형 범용 모델
                    "stepfun/step-3.5-flash:free",                   # Step 3.5 Flash (ctx 256K)
                    "google/gemma-3-27b-it:free",                    # Gemma 3 27B (ctx 131K)
                    "minimax/minimax-m2.5:free",                     # MiniMax M2.5 (ctx 196K)

                    # 경량/빠른 응답
                    "nvidia/nemotron-nano-9b-v2:free",               # Nemotron Nano 9B (ctx 128K)
                    "google/gemma-3-12b-it:free",                    # Gemma 3 12B (ctx 32K)
                    "google/gemma-3-4b-it:free",                     # Gemma 3 4B (ctx 32K)
                ],
                help="OpenRouter 무료 티어 모델 (일 50회 제한)"
        )
        if st.button("🔌 연결 테스트", use_container_width=True):
            client = OpenRouterClient(openrouter_api_key, model_name)
            ok, msg = client.check_connection()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    else:  # Claude
        claude_api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            help="Anthropic에서 발급받은 API Key"
        )
        claude_model = st.selectbox(
            "Claude 모델",
            ["sonnet", "opus", "haiku"],
            help="Claude Agent SDK 모델 (sonnet 권장)"
        )
        model_name = f"claude-{claude_model}"
        if st.button("🔌 연결 테스트", use_container_width=True):
            runner = ClaudeAgentRunner(llm_provider="Claude", api_key=claude_api_key, model=claude_model)
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
        help="Qwen3의 /think 모드 활성화 (더 정확하지만 느림)"
    )

    st.markdown("---")
    st.markdown("### 🔧 실행 모드")

    agent_mode = st.radio(
        "모드 선택",
        options=["🤖 에이전트 (계획→실행→검증)", "💬 채팅 (간단한 도구 사용)"],
        index=0,
        help="에이전트: To-do 계획 수립 후 순차 실행\n채팅: 자유로운 대화 + 필요시 도구 사용"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.status_log = []
            st.session_state.tool_log = []
            st.session_state.todo_items = []
            st.session_state.current_plan = None
            if st.session_state.agent:
                st.session_state.agent.reset_history()
            st.rerun()
    with col2:
        if st.button("📋 로그 지우기", use_container_width=True):
            st.session_state.status_log = []
            st.session_state.tool_log = []
            st.rerun()


# ──────────────────────────────────────────────
# 메인 화면 - 탭 (채팅 | 워크스페이스)
# ──────────────────────────────────────────────

st.markdown("# 🤖 DeepAssist")
st.caption(f"모델: `{model_name}` | 작업 디렉토리: `{display_path(st.session_state.working_dir)}`")

tab_chat, tab_ws = st.tabs(["💬 채팅", "📁 워크스페이스"])

# ──────────────────────────────────────────────
# 워크스페이스 탭
# ──────────────────────────────────────────────

# 워크스페이스 탭 세션 상태 초기화
if "editing_file" not in st.session_state:
    st.session_state["editing_file"] = None
if "_uploaded_hash" not in st.session_state:
    st.session_state["_uploaded_hash"] = None

with tab_ws:
    # ── 서버 연결 확인 ──
    server_ok = False
    ws_size, ws_quota, session_id = "?", "?", ""
    try:
        si = httpx.get(f"{FILE_SERVER_URL}/api/session", timeout=3)
        if si.status_code == 200:
            info = si.json()
            ws_path = info.get("workspace_path", "")
            if ws_path and os.path.isdir(ws_path):
                st.session_state.working_dir = ws_path
            ws_size    = info.get("workspace_size", "?")
            ws_quota   = info.get("workspace_quota", "?")
            session_id = info.get("session_id", "")
            server_ok  = True
        else:
            st.warning("⚠️ 파일 서버 응답 오류")
    except Exception:
        st.warning("⚠️ 파일 서버 미연결 — `python server.py` 실행 필요")

    if server_ok:
        # ── 헤더 정보 ──
        hdr_a, hdr_b = st.columns([5, 1])
        with hdr_a:
            st.markdown(f"**📁 워크스페이스**  ·  💾 `{ws_size}` / `{ws_quota}`  ·  🔑 `{session_id}`")
            st.caption("📌 파일당 최대 100MB · 전체 용량 최대 100MB · 허용 확장자: md, txt, py, json, yaml, csv, html, js, sh, env, toml, cfg, log")
        with hdr_b:
            if st.button("🔄 새로고침", key="ws_reload", use_container_width=True):
                st.rerun()

        st.divider()
        file_col, editor_col = st.columns([2, 3])

        # ── 파일 목록 (루트 기준 플랫 리스트) ──
        with file_col:
            try:
                lr = httpx.get(f"{FILE_SERVER_URL}/api/files/listdir",
                               params={"path": ""}, timeout=5)
                files = [i for i in lr.json().get("items", []) if i["type"] == "file"] \
                    if lr.status_code == 200 else []
            except Exception:
                files = []

            if files:
                st.caption(f"📄 파일 ({len(files)}개)")
                for fitem in files:
                    fname = fitem["name"]
                    fsize = fitem.get("size", "")
                    fmod  = fitem.get("modified", "")
                    fc1, fc2, fc3 = st.columns([5, 1, 1])
                    with fc1:
                        if st.button(f"📄 {fname}", key=f"opn_{fname}",
                                     help=f"{fsize}  ·  {fmod}"):
                            st.session_state["editing_file"] = fname
                    with fc2:
                        if st.button("⬇", key=f"dlb_{fname}", help="다운로드"):
                            st.session_state[f"_dl_{fname}"] = True
                        if st.session_state.pop(f"_dl_{fname}", False):
                            try:
                                dl = httpx.get(
                                    f"{FILE_SERVER_URL}/api/files/download/{fname}",
                                    timeout=10)
                                if dl.status_code == 200:
                                    st.download_button("💾", data=dl.content,
                                                       file_name=fname,
                                                       key=f"dls_{fname}")
                            except Exception:
                                pass
                    with fc3:
                        if st.button("🗑", key=f"del_{fname}"):
                            try:
                                r = httpx.delete(
                                    f"{FILE_SERVER_URL}/api/files/{fname}", timeout=5)
                                if r.status_code == 200:
                                    if st.session_state.get("editing_file") == fname:
                                        st.session_state["editing_file"] = None
                                    st.rerun()
                            except Exception:
                                pass
            else:
                st.caption("업로드된 파일이 없습니다.")

            # ── 하단 액션 ──
            st.divider()
            action = st.radio("작업", ["📄 새 파일", "⬆ 업로드"],
                              horizontal=True, label_visibility="collapsed",
                              key="ws_act")

            if action == "📄 새 파일":
                nfn = st.text_input("파일명", key="ws_nfn", placeholder="예: notes.md")
                if st.button("만들기", key="ws_mknfn", use_container_width=True) and nfn.strip():
                    try:
                        wr = httpx.post(f"{FILE_SERVER_URL}/api/files/write",
                                        json={"path": nfn.strip(), "content": ""},
                                        timeout=5)
                        if wr.status_code == 200:
                            st.session_state["editing_file"] = nfn.strip()
                            st.rerun()
                        else:
                            st.error(wr.json().get("detail", "생성 실패"))
                    except Exception as e:
                        st.error(f"❌ {e}")

            else:  # 업로드
                import hashlib as _hl
                uploaded = st.file_uploader(
                    "파일 선택 (최대 100MB)", label_visibility="visible",
                    type=[e.lstrip('.') for e in [
                        ".md", ".txt", ".py", ".json", ".yaml", ".yml",
                        ".csv", ".html", ".js", ".ts", ".sh", ".env",
                        ".toml", ".ini", ".cfg", ".log"]], key="ws_upl")
                if uploaded is not None:
                    fhash = _hl.md5(uploaded.getvalue()).hexdigest()
                    if fhash != st.session_state["_uploaded_hash"]:
                        try:
                            resp = httpx.post(
                                f"{FILE_SERVER_URL}/api/files/upload",
                                files={"file": (uploaded.name, uploaded.getvalue(),
                                               uploaded.type or "application/octet-stream")},
                                timeout=60)
                            if resp.status_code == 200:
                                st.session_state["_uploaded_hash"] = fhash
                                st.success(f"✅ '{uploaded.name}' 업로드 완료")
                                st.rerun()
                            else:
                                st.error(resp.json().get("detail", "업로드 실패"))
                        except Exception as e:
                            st.error(f"❌ {e}")

        # ── 파일 편집기 ──
        with editor_col:
            editing = st.session_state.get("editing_file")
            if editing:
                st.markdown(f"✏️ **편집:** `{editing}`")
                try:
                    rr = httpx.get(f"{FILE_SERVER_URL}/api/files/read/{editing}", timeout=5)
                    if rr.status_code == 200:
                        orig = rr.json().get("content", "")
                        edited = st.text_area("내용", value=orig, height=500,
                                              key=f"ed_{editing}",
                                              label_visibility="collapsed")
                        sc, cc = st.columns([1, 1])
                        with sc:
                            if st.button("💾 저장", key=f"sv_{editing}",
                                         use_container_width=True):
                                try:
                                    wr = httpx.post(f"{FILE_SERVER_URL}/api/files/write",
                                                    json={"path": editing, "content": edited},
                                                    timeout=10)
                                    if wr.status_code == 200:
                                        st.success("✅ 저장 완료")
                                    else:
                                        st.error(wr.json().get("detail", "저장 실패"))
                                except Exception as ex:
                                    st.error(f"❌ {ex}")
                        with cc:
                            if st.button("✖ 닫기", key=f"cl_{editing}",
                                         use_container_width=True):
                                st.session_state["editing_file"] = None
                                st.rerun()
                    elif rr.status_code == 400:
                        st.info("🔒 바이너리 파일은 텍스트 편집 불가")
                        if st.button("닫기", key="bin_cl"):
                            st.session_state["editing_file"] = None
                            st.rerun()
                except Exception as ex:
                    st.error(f"❌ {ex}")
            else:
                st.markdown("✏️ **파일 편집기**")
                st.caption("← 왼쪽에서 파일을 클릭하면 여기서 편집할 수 있습니다.")


# ──────────────────────────────────────────────
# 채팅 탭
# ──────────────────────────────────────────────

with tab_chat:
    # 2컬럼 레이아웃: 채팅 + 상태 패널
    chat_col, status_col = st.columns([3, 1])

    with status_col:
        todo_placeholder = st.empty()
        status_log_placeholder = st.empty()

# ──────────────────────────────────────────────
# 실시간 UI 렌더링 함수
# ──────────────────────────────────────────────


def update_todo_ui():
    """Todo 체크리스트를 상태 패널 상단에 렌더링"""
    with todo_placeholder.container():
        items = st.session_state.todo_items
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


def update_status_log_ui():
    with status_log_placeholder.container():
        st.markdown("### 📊 상태 로그")
        if st.session_state.status_log:
            log_text = "\n".join(st.session_state.status_log)
            with st.container(height=500):
                st.code(log_text, language="text")
        else:
            st.caption("실행 로그가 없습니다.")

# 초기 상태 렌더링
update_todo_ui()
update_status_log_ui()


# ──────────────────────────────────────────────
# 콜백 함수
# ──────────────────────────────────────────────
def on_status(msg: str):
    st.session_state.status_log.append(msg)
    update_status_log_ui()

def on_todo_update(items: list):
    st.session_state.todo_items = items
    update_todo_ui()

def on_tool_call(record: ToolCallRecord):
    st.session_state.tool_log.append({
        "name": record.tool_name,
        "args": record.arguments,
        "result": record.result,
        "time": record.timestamp
    })

def on_plan_update(plan: Plan):
    st.session_state.current_plan = {
        "goal": plan.goal,
        "tasks": [
            {"id": t.id, "desc": t.description, "status": t.status, "result": t.result}
            for t in plan.tasks
        ],
        "verified": plan.verified,
        "attempt": plan.attempt,
    }


# ──────────────────────────────────────────────
# 에이전트 초기화
# ──────────────────────────────────────────────
def _get_api_key() -> str:
    """현재 선택된 프로바이더의 API Key 반환"""
    if llm_provider == "Claude":
        return claude_api_key
    elif llm_provider == "Gemini API":
        return gemini_api_key
    elif llm_provider == "OpenRouter":
        return openrouter_api_key
    return "ollama_dummy_key"

def get_agent(is_agent_mode: bool):
    """에이전트 인스턴스 생성/반환: 에이전트 모드일 경우 무조건 ClaudeAgentRunner 활용"""
    if is_agent_mode:
        api_key_val = _get_api_key()
        model_val = claude_model if llm_provider == "Claude" else model_name

        runner = ClaudeAgentRunner(
            llm_provider=llm_provider,
            api_key=api_key_val,
            model=model_val,
            ollama_url=ollama_url if llm_provider == "Ollama" else None,
            max_turns=150,
            working_dir=st.session_state.working_dir,
            on_status=on_status,
            on_tool_call=on_tool_call,
            on_plan_update=on_plan_update,
        )
        runner.on_todo_update = on_todo_update
        return runner
    else:
        # 단순 챗
        if llm_provider == "Ollama":
            return OllamaClient(ollama_url, model_name)
        elif llm_provider == "Gemini API":
            return GeminiClient(gemini_api_key, model_name)
        elif llm_provider == "OpenRouter":
            return OpenRouterClient(openrouter_api_key, model_name)
        else:
            return ClaudeAgentRunner(
                llm_provider=llm_provider,
                api_key=claude_api_key,
                model=claude_model,
                working_dir=st.session_state.working_dir,
                max_turns=150,
            )


# ──────────────────────────────────────────────
# 채팅 UI
# ──────────────────────────────────────────────
with chat_col:
    # 메시지 표시
    for msg in st.session_state.messages:
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
    if not st.session_state.is_running:
        prompt = st.chat_input("무엇을 도와드릴까요?")
    else:
        st.chat_input("에이전트 실행 중...", disabled=True)
        prompt = None

    if prompt:
        st.session_state.is_running = True
        st.session_state.status_log = []
        st.session_state.tool_log = []
        st.session_state.todo_items = []
        st.session_state.current_plan = None

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        is_agent_mode = agent_mode.startswith("🤖")
        agent = get_agent(is_agent_mode)

        main_log_container = st.empty()
        live_logs: list[str] = []

        # 콜백 래핑은 ClaudeAgentRunner(에이전트 모드)에서만 적용
        if hasattr(agent, 'on_tool_call'):
            orig_on_tool_call = agent.on_tool_call
            def live_on_tool_call(record: ToolCallRecord):
                orig_on_tool_call(record)
                if llm_provider != "Claude":
                    live_logs.append(f"🔧 **{record.tool_name}** 호출 완료")
                    with main_log_container.container():
                        st.info("\n\n".join(live_logs[-8:]))
            agent.on_tool_call = live_on_tool_call

        if hasattr(agent, 'on_status'):
            orig_on_status = agent.on_status
            def live_on_status(msg: str):
                orig_on_status(msg)
                live_logs.append(msg)
                with main_log_container.container():
                    st.info("\n\n".join(live_logs[-8:]))
            agent.on_status = live_on_status

        if is_agent_mode:
            # ── 에이전트 모드: 스피너 + SDK 호출 ──
            with st.spinner("🤖 에이전트 작업 중..."):
                try:
                    response = agent.run(prompt)
                except Exception as e:
                    response = f"❌ 실행 중 오류가 발생했습니다:\n```\n{e}\n```"

            main_log_container.empty()

            # 상태 로그 파일 저장
            if st.session_state.status_log:
                import datetime
                now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = f"status_log_{now_str}.txt"
                log_path = os.path.join(st.session_state.working_dir, log_filename)
                try:
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(st.session_state.status_log))
                    st.session_state.status_log.append(f"📁 상태 로그 저장 완료: {log_filename}")
                except Exception as e:
                    st.session_state.status_log.append(f"❌ 상태 로그 저장 실패: {e}")

            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            # ── 단순 채팅 모드: 스트리밍 출력 ──
            with st.chat_message("assistant"):
                if hasattr(agent, 'run'):
                    # ClaudeAgentRunner (채팅 모드)
                    with st.spinner("💬 응답 생성 중..."):
                        try:
                            response = agent.chat(prompt)
                        except Exception as e:
                            response = f"❌ 오류:\n```\n{e}\n```"
                    st.markdown(response)
                else:
                    # OllamaClient / GeminiClient / OpenRouterClient — UI 스트리밍
                    msgs = [{"role": m["role"], "content": str(m.get("content", ""))} for m in st.session_state.messages]
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
        msg_data = {
            "role": "assistant",
            "content": response,
            "tool_calls": [
                {"name": t["name"], "args": t["args"], "result": t["result"][:200]}
                for t in st.session_state.tool_log
            ],
            "plan": st.session_state.current_plan
        }
        st.session_state.messages.append(msg_data)

        st.session_state.is_running = False
        st.rerun()

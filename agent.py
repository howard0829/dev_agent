"""
DeepAssist - Claude Agent SDK Runner
Claude Agent SDK 기반 자율 코딩 에이전트
"""

import logging
import os
import json
import re
import time
from typing import Dict, List, Optional, Callable

from models import Task, Plan, ToolCallRecord
from backend_strategy import select_strategy, BackendStrategy
from config import (
    OLLAMA_DEFAULT_URL, VLLM_DEFAULT_URL,
    AGENT_MAX_TURNS, DEEPASSIST_MD_MAX_SIZE,
)

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import (
        ClaudeSDKClient, ClaudeAgentOptions,
        AssistantMessage, ResultMessage,
        TextBlock, ToolUseBlock, ToolResultBlock,
    )
    CLAUDE_SDK_AVAILABLE = True
    _CLAUDE_IMPORT_ERROR = ""
except Exception as _claude_err:
    CLAUDE_SDK_AVAILABLE = False
    _CLAUDE_IMPORT_ERROR = str(_claude_err)


# 미완료 Task 자동 재시도 최대 횟수
MAX_CONTINUATIONS = 2

# SDK에 전달할 허용 도구 목록 (모델이 필요시 자동 선택)
ALLOWED_TOOLS = [
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "list_knowledge_dbs", "search_knowledge", "search_web_and_scrape",
]


class ClaudeAgentRunner:
    """Claude Agent SDK 기반 에이전트.

    SDK가 Plan 수립, 도구 호출, 검증을 전부 수행하므로
    별도의 Plan→Execute→Verify 루프 없이 동일한 콜백 인터페이스
    (on_status, on_tool_call)를 통해 app.py의 UI와 연동한다.

    백엔드 연결은 BackendStrategy에 위임하여
    Proxy(claude-code-proxy) / Ollama Native / vLLM 모드를 지원한다.
    """

    def __init__(
        self,
        llm_provider: str,
        api_key: str,
        model: str,
        ollama_url: str = None,
        vllm_url: str = None,
        working_dir: str = ".",
        max_turns: int = AGENT_MAX_TURNS,
        on_status: Callable = None,
        on_tool_call: Callable = None,
        on_plan_update: Callable = None,
        backend_mode: str = "auto",
    ):
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.model = model
        self.ollama_url = ollama_url or OLLAMA_DEFAULT_URL
        self.vllm_url = vllm_url or VLLM_DEFAULT_URL
        self.working_dir = os.path.abspath(working_dir)
        self.max_turns = max_turns
        self.backend_mode = backend_mode

        self.on_status = on_status or (lambda *a: None)
        self.on_tool_call = on_tool_call or (lambda *a: None)
        self.on_plan_update = on_plan_update or (lambda *a: None)
        self.on_todo_update: Callable = lambda *a: None
        self.on_agent_text: Callable = lambda *a: None

        self.tool_call_log: List[ToolCallRecord] = []
        self.conversation_history: List[Dict] = []
        self.current_plan: Optional[Plan] = None

    def _select_strategy(self) -> BackendStrategy:
        """프로바이더와 백엔드 모드에 따라 전략을 선택한다."""
        return select_strategy(
            llm_provider=self.llm_provider,
            backend_mode=self.backend_mode,
            model=self.model,
            api_key=self.api_key,
            ollama_url=self.ollama_url,
            vllm_url=self.vllm_url,
        )

    def check_connection(self) -> tuple:
        """연결 가능 여부를 사전 확인한다."""
        if not CLAUDE_SDK_AVAILABLE:
            return False, f"claude-agent-sdk 임포트 실패: {_CLAUDE_IMPORT_ERROR}"
        strategy = self._select_strategy()
        ok, msg = strategy.check()
        if not ok:
            return False, msg
        return True, f"Claude Agent SDK 준비됨 (Provider: {self.llm_provider}, Mode: {self.backend_mode}, Model: {self.model})"

    def run(self, prompt: str) -> str:
        """동기 래퍼: 별도 스레드에서 async SDK를 호출하고 결과를 반환"""
        if not CLAUDE_SDK_AVAILABLE:
            return f"Claude Agent SDK를 불러올 수 없습니다: {_CLAUDE_IMPORT_ERROR}\n\npip install claude-agent-sdk 로 설치해 주세요."
        import asyncio
        import threading
        import queue as queue_mod

        result_queue: queue_mod.Queue = queue_mod.Queue()
        done_event = threading.Event()

        def _thread_target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._async_run(prompt, result_queue)
                )
                result_queue.put(("__FINAL__", result))
            except Exception as e:
                logger.exception("에이전트 비동기 실행 중 오류")
                result_queue.put(("__ERROR__", str(e)))
            finally:
                loop.close()
                done_event.set()

        t = threading.Thread(target=_thread_target, daemon=True)
        t.start()

        final_result = ""
        while not done_event.is_set() or not result_queue.empty():
            try:
                event_type, data = result_queue.get(timeout=0.1)
                if event_type == "__FINAL__":
                    final_result = data
                    break
                elif event_type == "__ERROR__":
                    final_result = f"Claude 오류: {data}"
                    break
                elif event_type == "agent_text":
                    self.on_agent_text(data)
                elif event_type == "status":
                    self.on_status(data)
                elif event_type == "tool_call":
                    self.on_tool_call(data)
                elif event_type == "todo_update":
                    self.on_todo_update(data)
            except queue_mod.Empty:
                continue

        t.join(timeout=5)
        return final_result

    async def _async_run(self, prompt: str, event_queue) -> str:
        """실제 SDK 호출 (async)"""
        wd = self.working_dir

        # 시스템 프롬프트 구성
        system_prompt = (
            "You are an expert coding agent. Respond in Korean (한국어).\n"
            "\n"
            "## SANDBOX — WORKING DIRECTORY RESTRICTION\n"
            f"Your working directory is: {wd}\n"
            "ALL file operations (Read, Write, Edit, Glob, Grep) and Bash commands MUST target paths\n"
            f"that start with exactly `{wd}/` or equal `{wd}`.\n"
            "ABSOLUTE PROHIBITION:\n"
            f"- NEVER read, write, edit, list, or execute anything outside `{wd}/`.\n"
            "- NEVER access ~/.claude, ~/.config, ~/.ssh, ~/.bash_history, /tmp, /etc, or any parent directory.\n"
            "- NEVER use `cd ..` or relative paths that escape the working directory.\n"
            "- NEVER use symlinks, environment variable tricks, or shell expansions (~, $HOME) to escape the sandbox.\n"
            "- In Bash commands, NEVER run: `rm -rf /`, `chmod 777`, or any command that modifies files outside the sandbox.\n"
            "If a user explicitly asks to access a file outside the working directory, REFUSE and explain the restriction.\n"
            "\n"
            "## TOOL RULES\n"
            "1. You MUST USE TOOLS (Write, Edit, Bash, etc.) to create or alter files. DO NOT just output code blocks.\n"
            f"2. ALWAYS use ABSOLUTE PATHS starting with `{wd}/` for all file tool arguments.\n"
            "3. Before Write or Edit, you MUST call Read on that path first.\n"
        )

        # DeepAssist.md 자동 로드 (크기 제한 적용)
        md_path = os.path.join(wd, "DeepAssist.md")
        if os.path.exists(md_path):
            try:
                file_size = os.path.getsize(md_path)
                if file_size > DEEPASSIST_MD_MAX_SIZE:
                    logger.warning(
                        f"DeepAssist.md 크기({file_size}B)가 제한({DEEPASSIST_MD_MAX_SIZE}B)을 초과하여 건너뜁니다."
                    )
                else:
                    with open(md_path, "r", encoding="utf-8") as f:
                        system_prompt += f"\n## PROJECT GUIDELINES\n{f.read()}"
            except Exception as e:
                logger.warning(f"DeepAssist.md 로드 실패: {e}")

        # MCP 서버 설정
        mcp_dir = os.path.join(wd, ".claude")
        os.makedirs(mcp_dir, exist_ok=True)
        mcp_config_path = os.path.join(mcp_dir, "mcp.json")

        server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
        mcp_config = {
            "mcpServers": {
                "deepassist_tools": {
                    "command": "python",
                    "args": [server_path]
                }
            }
        }
        with open(mcp_config_path, "w", encoding="utf-8") as f:
            json.dump(mcp_config, f, indent=2)

        # SDK 옵션 구성
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=ALLOWED_TOOLS,
            model=None,
            permission_mode="acceptEdits",
        )

        # UI 표시용 경로 (symlink 해소 후 안전한 경로 추출)
        real_wd = os.path.realpath(wd)
        _display_wd = real_wd[real_wd.find("/workspaces/"):] if "/workspaces/" in real_wd else real_wd
        event_queue.put(("status", f"DeepAssist Agent 실행 중... (sandbox: {_display_wd})"))

        final_text = ""
        tool_counter = 0
        last_tool_record = None
        todo_items: List[dict] = []

        # Todo 리스트 강제화 + 진행 상황 설명 프롬프트
        forced_prompt = (
            "[System Instruction]\n"
            "## 작업 진행 규칙\n"
            "1. 작업을 시작하기 전에 반드시 구체적인 Todo List를 넘버링(1. 2. 3. ...) 형태로 먼저 작성하고 출력하세요.\n"
            "2. 각 Task를 시작할 때 '🔄 Task N 시작: <무엇을 할 것인지 한 줄 설명>' 형태로 출력하세요.\n"
            "3. 각 Task를 완료할 때 '✅ Task N 완료: <완료한 작업 요약>' 형태로 출력하세요.\n"
            "4. 도구를 호출하기 전에 왜 그 도구를 사용하는지 한 줄로 간단히 설명하세요.\n"
            "   예: '로그인 폼 컴포넌트를 작성합니다.' → Write 도구 호출\n"
            "5. 모든 Task를 완료한 후 반드시 마지막에 각 Task의 완료 여부를 점검하고, "
            "미완료 Task가 있으면 계속 수행하세요.\n"
            "6. MUST USE TOOLS: 코드를 작성하거나 수정할 때는 반드시 Write/Edit 등의 도구를 직접 호출하여 실제 파일 시스템에 저장하세요. "
            "절대로 마크다운 코드 블럭만 출력하고 파일 생성을 완료했다고 거짓말(Hallucinate)하지 마세요.\n\n"
            f"{prompt}"
        )

        async def _process_response(client, event_queue, todo_items, tool_counter, last_tool_record, final_text):
            """SDK 응답 메시지를 처리하는 내부 헬퍼"""
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text = block.text.strip()
                            if text:
                                final_text += block.text
                                event_queue.put(("agent_text", text))
                                event_queue.put(("status", text))
                                self._parse_markdown_todos(text, todo_items, event_queue)

                        elif isinstance(block, ToolUseBlock):
                            tool_counter += 1
                            tool_args = block.input if block.input else {}

                            # TodoWrite 도구 호출 → todo_update 이벤트로 변환
                            if block.name == "TodoWrite":
                                todos_raw = tool_args.get("todos", [])
                                todo_items.clear()
                                todo_items.extend([
                                    {"text": t.get("content", ""), "status": t.get("status", "pending")}
                                    for t in todos_raw
                                ])
                                event_queue.put(("todo_update", [t.copy() for t in todo_items]))
                                done = sum(1 for t in todo_items if t["status"] == "completed")
                                prog = sum(1 for t in todo_items if t["status"] == "in_progress")
                                event_queue.put(("status", f"Todo 업데이트 ({done} 완료 / {prog} 진행 중 / {len(todo_items)} 전체)"))
                                continue

                            summary = self._format_tool_summary(block.name, tool_args)

                            record = ToolCallRecord(
                                tool_name=block.name,
                                arguments=tool_args,
                                result="(실행 대기 중)",
                                timestamp=time.time(),
                            )
                            self.tool_call_log.append(record)
                            last_tool_record = record
                            event_queue.put(("tool_call", record))
                            event_queue.put(("status", f"[{tool_counter}] {block.name} — {summary}"))

                        elif isinstance(block, ToolResultBlock):
                            result_text = ""
                            if hasattr(block, 'content'):
                                if isinstance(block.content, str):
                                    result_text = block.content
                                elif isinstance(block.content, list):
                                    for item in block.content:
                                        if hasattr(item, 'text'):
                                            result_text += item.text
                                        elif isinstance(item, dict) and 'text' in item:
                                            result_text += item['text']

                            if last_tool_record:
                                last_tool_record.result = result_text[:2000] if result_text else "(완료)"
                                event_queue.put(("tool_call", last_tool_record))

                            if result_text:
                                event_queue.put(("status", f"   -> 결과: {result_text.strip()}"))
                            else:
                                event_queue.put(("status", "   -> 완료"))

                elif isinstance(msg, ResultMessage):
                    if hasattr(msg, 'result') and msg.result:
                        final_text = msg.result
                    if hasattr(msg, 'usage') and msg.usage:
                        usage = msg.usage
                        event_queue.put(("status",
                            f"토큰: 입력={getattr(usage, 'input_tokens', '?')} "
                            f"출력={getattr(usage, 'output_tokens', '?')}"
                        ))

            return tool_counter, last_tool_record, final_text

        # 백엔드 전략 선택 및 활성화
        strategy = self._select_strategy()

        try:
            strategy.activate(event_queue)

            async with ClaudeSDKClient(options=options) as client:
                # 1차 실행
                await client.query(forced_prompt)
                tool_counter, last_tool_record, final_text = await _process_response(
                    client, event_queue, todo_items, tool_counter, last_tool_record, final_text
                )

                # 미완료 Task 점검 및 자동 재시도
                for attempt in range(MAX_CONTINUATIONS):
                    incomplete = [
                        (i, t) for i, t in enumerate(todo_items)
                        if t["status"] in ("pending", "in_progress")
                    ]
                    if not incomplete:
                        break

                    incomplete_list = "\n".join(
                        f"  {i + 1}. {t['text']} (상태: {t['status']})"
                        for i, t in incomplete
                    )
                    event_queue.put(("status",
                        f"⚠️ 미완료 Task {len(incomplete)}개 감지 — "
                        f"추가 수행 중 (시도 {attempt + 1}/{MAX_CONTINUATIONS})"
                    ))
                    event_queue.put(("agent_text",
                        f"⚠️ 미완료 Task {len(incomplete)}개 감지 — 계속 수행합니다..."
                    ))

                    cont_prompt = (
                        f"⚠️ 다음 {len(incomplete)}개의 Task가 아직 미완료 상태입니다. "
                        f"이어서 수행하세요:\n{incomplete_list}\n\n"
                        "각 Task를 시작할 때 '🔄 Task N 시작: ...' 형태로, "
                        "완료할 때 '✅ Task N 완료: ...' 형태로 출력하세요."
                    )
                    await client.query(cont_prompt)
                    tool_counter, last_tool_record, final_text = await _process_response(
                        client, event_queue, todo_items, tool_counter, last_tool_record, final_text
                    )

        finally:
            strategy.cleanup(event_queue)

        # 최종 완료 상태 보고
        if todo_items:
            total = len(todo_items)
            done = sum(1 for t in todo_items if t["status"] == "completed")
            incomplete = [t for t in todo_items if t["status"] != "completed"]

            if incomplete:
                incomplete_summary = "\n".join(f"  - {t['text']}" for t in incomplete)
                event_queue.put(("status",
                    f"⚠️ 미완료 Task {len(incomplete)}/{total}개:\n{incomplete_summary}"
                ))
                event_queue.put(("agent_text",
                    f"⚠️ **미완료 Task {len(incomplete)}/{total}개:**\n{incomplete_summary}"
                ))
            else:
                event_queue.put(("status", f"✅ 모든 Task 완료 ({done}/{total})"))
                event_queue.put(("agent_text", f"✅ **모든 Task 완료 ({done}/{total})**"))

            event_queue.put(("todo_update", [t.copy() for t in todo_items]))

        event_queue.put(("status", f"Claude Agent 작업 완료 (도구 {tool_counter}회 호출)"))
        return final_text

    @staticmethod
    def _short_path(path: str) -> str:
        """절대 경로에서 /workspaces/ 이후만 표시"""
        idx = path.find("/workspaces/")
        return path[idx:] if idx != -1 else path

    @staticmethod
    def _parse_markdown_todos(text: str, todo_items: list, event_queue) -> None:
        """텍스트에서 넘버링 Todo, 시작/완료 마커를 파싱하여 todo_items를 갱신"""
        changed = False
        for line in text.split("\n"):
            line_s = line.strip()
            # 넘버링 Todo 항목 파싱 (1. 항목, 2. 항목, ...)
            m = re.match(r'^(\d+)\.\s+(.+)', line_s)
            if m:
                item_text = m.group(2).strip()
                if not any(t["text"] == item_text for t in todo_items):
                    todo_items.append({"text": item_text, "status": "pending"})
                    changed = True
                continue
            # 🔄 Task N 시작 마커 파싱
            m = re.match(r'^🔄\s*[Tt]ask\s*(\d+)\s*시작[:\s]*(.*)', line_s)
            if m:
                task_num = int(m.group(1))
                idx = task_num - 1
                if 0 <= idx < len(todo_items) and todo_items[idx]["status"] != "completed":
                    todo_items[idx]["status"] = "in_progress"
                    changed = True
                    event_queue.put(("status", f"🔄 Task {task_num} 시작: {todo_items[idx]['text']}"))
                continue
            # ✅ Task N 완료 마커 파싱
            m = re.match(r'^✅\s*[Tt]ask\s*(\d+)\s*완료[:\s]*(.*)', line_s)
            if m:
                task_num = int(m.group(1))
                idx = task_num - 1
                if 0 <= idx < len(todo_items) and todo_items[idx]["status"] != "completed":
                    todo_items[idx]["status"] = "completed"
                    changed = True
                    event_queue.put(("status", f"✅ Task {task_num} 완료: {todo_items[idx]['text']}"))
                continue
        if changed:
            event_queue.put(("todo_update", [t.copy() for t in todo_items]))

    @staticmethod
    def _format_tool_summary(tool_name: str, args: dict) -> str:
        """도구별 맞춤 요약 문자열 생성"""
        def _short(p: str) -> str:
            idx = p.find("/workspaces/")
            return p[idx:] if idx != -1 else p

        if tool_name == "Bash":
            cmd = args.get("command", "")
            return f"`{cmd}`" if cmd else "(빈 명령)"
        elif tool_name in ("Read", "Write", "Edit"):
            path = _short(args.get("file_path", args.get("path", "")))
            if tool_name == "Write":
                content = args.get("content", "")
                lines = content.count('\n') + 1 if content else 0
                return f"{path} ({lines}줄)"
            elif tool_name == "Edit":
                old = args.get("old_string", "")[:30]
                new = args.get("new_string", "")[:30]
                return f"{path} (교체: '{old}' -> '{new}')"
            return path
        elif tool_name in ("Glob", "Grep"):
            pattern = args.get("pattern", args.get("query", ""))
            return f"패턴: {pattern}"
        else:
            parts = [f"{k}={v}" for k, v in args.items()]
            return ", ".join(parts)

    def chat(self, prompt: str) -> str:
        """단순 대화 모드 (run과 동일하게 SDK 호출)"""
        return self.run(prompt)

    def reset_history(self):
        """대화 히스토리 초기화"""
        self.conversation_history = []
        self.tool_call_log = []
        self.current_plan = None

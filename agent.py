"""
DeepAssist - Claude Agent SDK Runner
Claude Agent SDK 기반 자율 코딩 에이전트
"""

import os
import json
import re
import time
from typing import Dict, List, Optional, Callable

from models import Task, Plan, ToolCallRecord

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
    """

    def __init__(
        self,
        llm_provider: str,
        api_key: str,
        model: str,
        ollama_url: str = None,
        working_dir: str = ".",
        max_turns: int = 150,
        on_status: Callable = None,
        on_tool_call: Callable = None,
        on_plan_update: Callable = None,
    ):
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.model = model
        self.ollama_url = ollama_url
        self.working_dir = os.path.abspath(working_dir)
        self.max_turns = max_turns

        self.on_status = on_status or (lambda *a: None)
        self.on_tool_call = on_tool_call or (lambda *a: None)
        self.on_plan_update = on_plan_update or (lambda *a: None)
        self.on_todo_update: Callable = lambda *a: None

        self.tool_call_log: List[ToolCallRecord] = []
        self.conversation_history: List[Dict] = []
        self.current_plan: Optional[Plan] = None

    def check_connection(self) -> tuple:
        if self.llm_provider == "Claude" and not self.api_key:
            return False, "ANTHROPIC_API_KEY가 설정되지 않았습니다."
        if not CLAUDE_SDK_AVAILABLE:
            return False, f"claude-agent-sdk 임포트 실패: {_CLAUDE_IMPORT_ERROR}"
        return True, f"Claude Agent SDK 준비됨 (Provider: {self.llm_provider}, Model: {self.model})"

    def run(self, prompt: str) -> str:
        """동기 래퍼: 별도 스레드에서 async SDK를 호출하고 결과를 반환"""
        if not CLAUDE_SDK_AVAILABLE:
            return f"Claude Agent SDK를 불러올 수 없습니다: {_CLAUDE_IMPORT_ERROR}\n\npip install claude-agent-sdk 로 설치해 주세요."
        import asyncio
        import threading
        import queue as queue_mod

        result_queue = queue_mod.Queue()

        def _thread_target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._async_run(prompt, result_queue)
                )
                result_queue.put(("__FINAL__", result))
            except Exception as e:
                result_queue.put(("__ERROR__", str(e)))
            finally:
                loop.close()

        t = threading.Thread(target=_thread_target, daemon=True)
        t.start()

        final_result = ""
        while True:
            try:
                event_type, data = result_queue.get(timeout=0.2)
                if event_type == "__FINAL__":
                    final_result = data
                    break
                elif event_type == "__ERROR__":
                    final_result = f"Claude 오류: {data}"
                    break
                elif event_type == "status":
                    self.on_status(data)
                elif event_type == "tool_call":
                    self.on_tool_call(data)
                elif event_type == "todo_update":
                    self.on_todo_update(data)
            except queue_mod.Empty:
                if not t.is_alive():
                    break

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

        # DeepAssist.md 자동 로드
        md_path = os.path.join(wd, "DeepAssist.md")
        if os.path.exists(md_path):
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    system_prompt += f"\n## PROJECT GUIDELINES\n{f.read()}"
            except Exception:
                pass

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
            model=self.model if self.llm_provider == "Claude" else None,
            permission_mode="acceptEdits",
        )

        # UI 표시용 경로: /workspaces/ 이후만 노출
        _display_wd = wd[wd.find("/workspaces/"):] if "/workspaces/" in wd else wd
        event_queue.put(("status", f"DeepAssist Agent 실행 중... (sandbox: {_display_wd})"))

        final_text = ""
        tool_counter = 0
        last_tool_record = None
        proxy_process = None
        todo_items: List[dict] = []  # [{"text": "...", "done": bool}]

        # Todo 리스트 강제화 프롬프트
        forced_prompt = (
            "[System Instruction: 작업을 시작하기 전에 반드시 구체적인 Todo List를 넘버링(1. 2. 3. ...) 형태로 먼저 작성하고 출력하세요. "
            "각 Task를 완료할 때마다 반드시 '✅ Task N 완료: <작업 내용>' 형태로 출력하세요. "
            "그 후 첫 번째 항목부터 실행하세요. MUST USE TOOLS: 코드를 작성하거나 수정할 때는 반드시 'Write' 또는 'Edit' 등의 도구(Tool)를 직접 호출하여 실제 파일 시스템에 저장하세요. "
            "절대로 마크다운 코드 블럭만 출력하고 파일 생성을 완료했다고 거짓말(Hallucinate)하지 마세요.]\n\n"
            f"{prompt}"
        )

        try:
            if self.llm_provider != "Claude":
                proxy_process = self._start_litellm_proxy(event_queue)
            else:
                if "ANTHROPIC_BASE_URL" in os.environ:
                    del os.environ["ANTHROPIC_BASE_URL"]
                os.environ["ANTHROPIC_API_KEY"] = self.api_key

            async with ClaudeSDKClient(options=options) as client:
                await client.query(forced_prompt)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                text = block.text.strip()
                                if text:
                                    final_text += block.text
                                    event_queue.put(("status", text))

                                    # 마크다운 체크박스 파싱 (Gemini/OpenRouter 등 TodoWrite 미지원 모델용)
                                    self._parse_markdown_todos(text, todo_items, event_queue)

                            elif isinstance(block, ToolUseBlock):
                                tool_counter += 1
                                tool_args = block.input if block.input else {}

                                # Todo: 도구 시작 시 in_progress → completed 전환
                                if todo_items and block.name not in ("TodoWrite",):
                                    changed = False
                                    completed_idx = -1
                                    for i, t in enumerate(todo_items):
                                        if t["status"] == "in_progress":
                                            t["status"] = "completed"
                                            completed_idx = i
                                            changed = True
                                            break
                                    if changed:
                                        if completed_idx >= 0:
                                            event_queue.put(("status", f"✅ Task {completed_idx + 1} 완료: {todo_items[completed_idx]['text']}"))
                                        # 다음 pending → in_progress
                                        for t in todo_items:
                                            if t["status"] == "pending":
                                                t["status"] = "in_progress"
                                                break
                                        event_queue.put(("todo_update", [t.copy() for t in todo_items]))

                                # TodoWrite 도구 호출 → todo_update 이벤트로 변환
                                if block.name == "TodoWrite":
                                    todos_raw = tool_args.get("todos", [])
                                    todo_items = [
                                        {"text": t.get("content", ""), "status": t.get("status", "pending")}
                                        for t in todos_raw
                                    ]
                                    event_queue.put(("todo_update", todo_items))
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

        finally:
            if proxy_process:
                proxy_process.terminate()
            # LiteLLM config 임시 파일 정리
            cfg_path = getattr(self, "_litellm_config_path", None)
            if cfg_path and os.path.exists(cfg_path):
                try:
                    os.unlink(cfg_path)
                except OSError:
                    pass

        # 남은 in_progress/pending 항목을 모두 completed로 마무리
        if todo_items:
            for t in todo_items:
                if t["status"] in ("pending", "in_progress"):
                    t["status"] = "completed"
            event_queue.put(("todo_update", [t.copy() for t in todo_items]))

        event_queue.put(("status", f"Claude Agent 작업 완료 (도구 {tool_counter}회 호출)"))
        return final_text

    def _start_litellm_proxy(self, event_queue):
        """Ollama/Gemini/OpenRouter용 LiteLLM 프록시 프로세스 시작"""
        import subprocess
        import shutil
        import tempfile
        import yaml

        event_queue.put(("status", f"{self.llm_provider} 연결을 위한 LiteLLM 프록시 시작 중..."))

        # 프로바이더별 LiteLLM 모델 파라미터 구성
        litellm_params = {"model": ""}

        if self.llm_provider == "Ollama":
            litellm_params["model"] = f"ollama/{self.model}"
            if self.ollama_url:
                litellm_params["api_base"] = self.ollama_url
        elif self.llm_provider == "OpenRouter":
            if self.model.startswith("openrouter/"):
                litellm_params["model"] = self.model
            else:
                litellm_params["model"] = f"openrouter/{self.model}"
            litellm_params["api_key"] = self.api_key
        else:  # Gemini
            litellm_params["model"] = f"gemini/{self.model}"
            litellm_params["api_key"] = self.api_key

        # Claude CLI가 보내는 모델명을 실제 모델로 매핑하는 config 생성
        # CLI는 기본 모델명(claude-sonnet-4-6 등)으로 요청하므로
        # 와일드카드 매핑으로 모든 요청을 지정 모델로 라우팅
        config = {
            "model_list": [
                {
                    "model_name": "claude-sonnet-4-6",
                    "litellm_params": dict(litellm_params),
                },
                {
                    "model_name": "claude-sonnet-4-20250514",
                    "litellm_params": dict(litellm_params),
                },
                {
                    "model_name": "claude-haiku-4-5",
                    "litellm_params": dict(litellm_params),
                },
                {
                    "model_name": "claude-opus-4-6",
                    "litellm_params": dict(litellm_params),
                },
            ],
            "litellm_settings": {
                "drop_params": True,
            },
        }

        # 임시 config 파일 생성
        config_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="litellm_cfg_", delete=False
        )
        yaml.dump(config, config_file, default_flow_style=False)
        config_file.close()

        os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:4000"
        os.environ["ANTHROPIC_API_KEY"] = self.api_key if self.api_key else "dummy"

        # 프로바이더별 환경변수 (config에서 api_key를 직접 넣지 않는 경우 fallback)
        if self.llm_provider == "Ollama" and self.ollama_url:
            os.environ["OLLAMA_API_BASE"] = self.ollama_url
        elif self.llm_provider in ["Gemini API", "Gemini"]:
            os.environ["GEMINI_API_KEY"] = self.api_key
        elif self.llm_provider == "OpenRouter":
            os.environ["OPENROUTER_API_KEY"] = self.api_key

        litellm_path = shutil.which("litellm") or "litellm"
        cmd = [litellm_path, "--config", config_file.name, "--port", "4000"]

        proxy_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._litellm_config_path = config_file.name  # cleanup용 보관
        time.sleep(4)
        return proxy_process

    @staticmethod
    def _short_path(path: str) -> str:
        """절대 경로에서 /workspaces/ 이후만 표시"""
        idx = path.find("/workspaces/")
        return path[idx:] if idx != -1 else path

    @staticmethod
    def _parse_markdown_todos(text: str, todo_items: list, event_queue) -> None:
        """텍스트에서 넘버링 Todo와 완료 메시지를 파싱하여 todo_items를 갱신"""
        changed = False
        for line in text.split("\n"):
            line_s = line.strip()
            # 넘버링 항목: 1. ... / 2. ... (Todo 리스트 파싱)
            m = re.match(r'^(\d+)\.\s+(.+)', line_s)
            if m:
                item_text = m.group(2).strip()
                if not any(t["text"] == item_text for t in todo_items):
                    todo_items.append({"text": item_text, "status": "pending"})
                    changed = True
                continue
            # 완료 메시지: ✅ Task N 완료: ...
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

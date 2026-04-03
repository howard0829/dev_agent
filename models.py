"""
DeepAssist - 데이터 모델
에이전트 실행에 사용되는 핵심 데이터 구조 및 타입 정의
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from typing_extensions import TypedDict


# ──────────────────────────────────────────────
# TypedDict — 딕셔너리 기반 설정의 타입 안전성 보장
# ──────────────────────────────────────────────

class ProviderConfig(TypedDict, total=False):
    """LLM 프로바이더 설정 딕셔너리 타입.

    core/sidebar.py의 render_llm_sidebar() 반환값,
    앱 page 모듈의 render_sidebar() 반환값에 사용.
    """
    llm_provider: str
    model_name: str
    api_key: str
    ollama_url: str
    vllm_url: str
    enable_thinking: bool
    agent_mode: str
    backend_mode: str
    # TestMancer 전용 (optional)
    test_framework: str
    test_target: str
    test_mode: str


class CallbackSet(TypedDict, total=False):
    """콜백 함수 세트 타입.

    core/chat_ui.py의 _make_callbacks() 반환값에 사용.
    """
    on_status: Any       # Callable[[str], None]
    on_todo_update: Any  # Callable[[list], None]
    on_tool_call: Any    # Callable[[ToolCallRecord], None]
    on_plan_update: Any  # Callable[[Plan], None]
    update_todo_ui: Any  # Callable[[], None]
    update_status_log_ui: Any  # Callable[[], None]


# ──────────────────────────────────────────────
# 데이터클래스 — 에이전트 실행 데이터 구조
# ──────────────────────────────────────────────

@dataclass
class Task:
    id: int
    description: str
    status: str = "pending"     # pending / running / done / failed
    result: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """직렬화용 딕셔너리 변환"""
        return asdict(self)


@dataclass
class Plan:
    goal: str
    tasks: List[Task] = field(default_factory=list)
    verified: bool = False
    attempt: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """직렬화용 딕셔너리 변환 (세션 상태 저장용)"""
        return {
            "goal": self.goal,
            "tasks": [
                {"id": t.id, "desc": t.description, "status": t.status, "result": t.result}
                for t in self.tasks
            ],
            "verified": self.verified,
            "attempt": self.attempt,
        }


@dataclass
class ToolCallRecord:
    """도구 호출 기록"""
    tool_name: str
    arguments: Dict[str, Any]
    result: str
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """직렬화용 딕셔너리 변환"""
        return {
            "name": self.tool_name,
            "args": self.arguments,
            "result": self.result,
            "time": self.timestamp,
        }

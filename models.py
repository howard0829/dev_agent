"""
DeepAssist - 데이터 모델
에이전트 실행에 사용되는 핵심 데이터 구조
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class Task:
    id: int
    description: str
    status: str = "pending"     # pending / running / done / failed
    result: str = ""

@dataclass
class Plan:
    goal: str
    tasks: List[Task] = field(default_factory=list)
    verified: bool = False
    attempt: int = 0

@dataclass
class ToolCallRecord:
    """도구 호출 기록"""
    tool_name: str
    arguments: Dict[str, Any]
    result: str
    timestamp: float = 0.0

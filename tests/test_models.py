"""
tests/test_models.py — Task/Plan/ToolCallRecord 직렬화 테스트
"""

from models import Task, Plan, ToolCallRecord


def test_task_to_dict():
    """Task.to_dict()가 올바른 딕셔너리를 반환하는지 확인"""
    t = Task(id=1, description="테스트 작업", status="running", result="완료")
    d = t.to_dict()

    assert d["id"] == 1
    assert d["description"] == "테스트 작업"
    assert d["status"] == "running"
    assert d["result"] == "완료"


def test_task_default_values():
    """Task 기본값 확인"""
    t = Task(id=0, description="기본")
    assert t.status == "pending"
    assert t.result == ""


def test_plan_to_dict():
    """Plan.to_dict()가 tasks를 포함한 올바른 구조를 반환하는지 확인"""
    tasks = [
        Task(id=1, description="단계 1", status="done", result="OK"),
        Task(id=2, description="단계 2", status="pending"),
    ]
    plan = Plan(goal="테스트 계획", tasks=tasks, verified=True, attempt=2)
    d = plan.to_dict()

    assert d["goal"] == "테스트 계획"
    assert d["verified"] is True
    assert d["attempt"] == 2
    assert len(d["tasks"]) == 2
    assert d["tasks"][0]["desc"] == "단계 1"
    assert d["tasks"][0]["status"] == "done"
    assert d["tasks"][1]["id"] == 2


def test_plan_empty_tasks():
    """빈 tasks 리스트의 Plan 직렬화"""
    plan = Plan(goal="빈 계획")
    d = plan.to_dict()

    assert d["tasks"] == []
    assert d["verified"] is False
    assert d["attempt"] == 0


def test_tool_call_record_to_dict():
    """ToolCallRecord.to_dict()가 올바른 키로 매핑되는지 확인"""
    record = ToolCallRecord(
        tool_name="read_file",
        arguments={"path": "/tmp/test.py"},
        result="파일 내용...",
        timestamp=1234567890.0,
    )
    d = record.to_dict()

    assert d["name"] == "read_file"
    assert d["args"] == {"path": "/tmp/test.py"}
    assert d["result"] == "파일 내용..."
    assert d["time"] == 1234567890.0


def test_tool_call_record_default_timestamp():
    """ToolCallRecord 기본 timestamp 값"""
    record = ToolCallRecord(tool_name="test", arguments={}, result="ok")
    assert record.timestamp == 0.0

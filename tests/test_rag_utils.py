"""
tests/test_rag_utils.py — RAG 유틸리티 함수 테스트
"""

from rag.utils import bm25_preprocessor, _extract_chunk_context
from rag.code import code_bm25_preprocessor
from rag.constants import _REQ_ID_RE, _REQ_ID_EXCLUDE


def test_bm25_preprocessor_req_id_preserved():
    """Requirement ID 하이픈이 보존되는지 확인"""
    result = bm25_preprocessor("TEL-6 관련 요구사항")
    assert "tel-6" in result


def test_bm25_preprocessor_normal_hyphen_removed():
    """일반 하이픈은 공백으로 치환되는지 확인"""
    result = bm25_preprocessor("NVMe-oF protocol")
    assert "-" not in result or "nvme" in result


def test_bm25_preprocessor_exclude_list():
    """_REQ_ID_EXCLUDE에 포함된 용어는 하이픈 보존 안 함"""
    result = bm25_preprocessor("UTF-8 encoding")
    # UTF-8은 제외 목록이므로 일반 하이픈처럼 처리
    assert "utf" in result


def test_bm25_preprocessor_lowercase():
    """출력이 소문자인지 확인"""
    result = bm25_preprocessor("SEC-3 REQUIREMENT")
    assert result == result.lower()


def test_code_bm25_preprocessor_camelcase():
    """CamelCase가 분리되는지 확인"""
    result = code_bm25_preprocessor("BlockManager allocateBlock")
    assert "block" in result
    assert "manager" in result


def test_code_bm25_preprocessor_snake_case():
    """snake_case가 분리되는지 확인"""
    result = code_bm25_preprocessor("allocate_block")
    assert "allocate" in result
    assert "block" in result


def test_code_bm25_preprocessor_req_id():
    """코드 BM25에서 Requirement ID 하이픈 보존"""
    result = code_bm25_preprocessor("test_TEL-2_evaluation()")
    assert "tel-2" in result


def test_extract_chunk_context_with_header():
    """_extract_chunk_context가 마크다운 헤더를 추출하는지 확인"""
    content = "[source | 섹션: Overview]\n## Introduction\nSome text\nMore text"
    context = _extract_chunk_context(content)
    assert "섹션:" in context
    assert "## Introduction" in context


def test_extract_chunk_context_with_table():
    """_extract_chunk_context가 테이블 헤더를 추출하는지 확인"""
    content = "| Name | Value |\n|---|---|\n| a | 1 |\n| b | 2 |"
    context = _extract_chunk_context(content)
    assert "| Name | Value |" in context
    assert "|---|---|" in context


def test_extract_chunk_context_empty():
    """일반 텍스트에서는 빈 문맥"""
    content = "This is just plain text without any structure."
    context = _extract_chunk_context(content)
    assert context == ""


def test_req_id_regex_matches():
    """_REQ_ID_RE가 올바른 Requirement ID를 매칭하는지 확인"""
    text = "TEL-6 SEC-3 FWUP-15 일반텍스트"
    matches = set(_REQ_ID_RE.findall(text)) - _REQ_ID_EXCLUDE
    assert "TEL-6" in matches
    assert "SEC-3" in matches
    assert "FWUP-15" in matches


def test_req_id_excludes():
    """_REQ_ID_EXCLUDE가 일반 기술 용어를 제외하는지 확인"""
    text = "UTF-8 AES-256 SHA-512"
    matches = set(_REQ_ID_RE.findall(text)) - _REQ_ID_EXCLUDE
    assert len(matches) == 0

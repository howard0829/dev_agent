"""
rag/__main__.py — MarkdownRAG / CodeRAG 테스트 실행

사용법:
    python -m rag             → 전체 테스트
    python -m rag markdown    → 마크다운 RAG만
    python -m rag code        → 코드 RAG만
"""

import sys
from typing import List

from langchain_core.documents import Document

from rag.markdown import MarkdownRAG
from rag.code import CodeRAG


def _print_results(tag: str, query: str, docs: List[Document]) -> None:
    """검색 결과를 포맷팅하여 출력"""
    print(f"\n{'='*60}")
    print(f"[{tag}] 🔍 쿼리: {query}")
    print('='*60)
    for i, doc in enumerate(docs):
        meta = doc.metadata
        chunk_type = meta.get("chunk_type", "")

        section = meta.get("section", "")
        page = meta.get("page", "")

        hierarchy = meta.get("hierarchy", "")
        signature = meta.get("signature", "")
        language = meta.get("language", "")
        line_range = meta.get("line_range", "")

        source = meta.get("source", "Unknown")
        req_ids = meta.get("requirement_ids", [])
        req_info = f" | IDs: {', '.join(req_ids[:20])}" if req_ids else ""
        if len(req_ids) > 20:
            req_info += f" (+{len(req_ids)-20})"

        if chunk_type in ("file_summary", "function", "class", "subchunk", "declarations"):
            project = meta.get("project", "")
            info_parts = [f"결과 {i+1}"]
            if project:
                info_parts.append(f"프로젝트: {project}")
            if language:
                info_parts.append(f"언어: {language}")
            if chunk_type:
                info_parts.append(f"유형: {chunk_type}")
            if line_range:
                info_parts.append(f"라인: {line_range[0]}-{line_range[1]}")
            if hierarchy:
                info_parts.append(f"계층: {hierarchy}")
            info_parts.append(f"파일: {source}")
            print(f"  {' | '.join(info_parts)}{req_info}")
            if signature and chunk_type != "file_summary":
                print(f"  시그니처: {signature}")
        else:
            doc_name = meta.get("doc_name", "")
            doc_info = f" | 문서: {doc_name}" if doc_name else ""
            page_info = f" | 페이지: {page}" if page else ""
            section_info = f" | 섹션: {section}" if section else ""
            print(f"  결과 {i+1}{doc_info}{page_info}{section_info}{req_info}")

        content = doc.page_content
        if len(content) > 2000:
            content = content[:1000] + f"\n... (중략, 총 {len(content)}자) ...\n" + content[-500:]
        print(f"  내용:\n{content}")
        print(f"  {'-'*56}")


def _run_test_suite(rag, label: str, queries: List[dict]) -> None:
    """테스트 쿼리 목록을 순차 실행하고 결과 출력"""
    for i, q in enumerate(queries, 1):
        tag = f"{label} Test {i} - {q['tag']}"
        docs = rag.retrieve(q["query"], top_k=q.get("top_k", 2))
        _print_results(tag, q["query"], docs)


# ══════════════════════════════════════════════════════════════
# __main__ — MarkdownRAG / CodeRAG 테스트 실행
# ══════════════════════════════════════════════════════════════

TESTS = [
    # ── 마크다운 RAG 테스트 예시 ──────────────────────────────
    # 아래 source/db_path 경로를 실제 환경에 맞게 변경하여 사용하세요.
    # {
    #     "type": "markdown",
    #     "label": "NVMe 2.3 Base Spec",
    #     "source": "/path/to/your/markdown/document.md",
    #     "db_path": "/path/to/knowledge/nvme23",
    #     "queries": [
    #         {"tag": "의미 검색",       "query": "큐(Queue)의 제출 및 완료 메커니즘은 어떻게 동작하나요?"},
    #         {"tag": "약어: RESERVS",  "query": "RESERVS"},
    #     ],
    # },
    # ── 코드 RAG 테스트 예시 ──────────────────────────────────
    # {
    #     "type": "code",
    #     "label": "NVMe Test Framework",
    #     "source": "/path/to/your/code/project",
    #     "db_path": "/path/to/knowledge/nvme_test_code",
    #     "queries": [
    #         {"tag": "Req ID 검색",   "query": "TEL-2 관련 평가를 진행하고 싶어"},
    #         {"tag": "심볼 검색",      "query": "allocate_block"},
    #     ],
    # },
]

# CLI 인자 파싱
mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
valid_modes = {"all", "markdown", "code"}
if mode not in valid_modes:
    print(f"사용법: python -m rag [{' | '.join(sorted(valid_modes))}]")
    print(f"  all      : 전체 테스트 (기본값)")
    print(f"  markdown : 마크다운 RAG만 테스트")
    print(f"  code     : 코드 RAG만 테스트")
    sys.exit(1)

filtered_tests = [t for t in TESTS if mode == "all" or t["type"] == mode]
if not filtered_tests:
    print(f"⚠️ '{mode}' 유형의 테스트 설정이 없습니다. TESTS 리스트에 항목을 추가하세요.")
    sys.exit(0)

try:
    for test_cfg in filtered_tests:
        test_type = test_cfg["type"]
        label = test_cfg["label"]
        source = test_cfg["source"]
        db_path = test_cfg["db_path"]

        print(f"\n{'='*60}")
        print(f"🚀 [{test_type.upper()}] {label} RAG 구축/로드")
        print(f"{'='*60}")

        if test_type == "markdown":
            rag = MarkdownRAG(db_store_path=db_path)
        elif test_type == "code":
            rag = CodeRAG(db_store_path=db_path)
        else:
            print(f"⚠️ 알 수 없는 유형: {test_type}")
            continue

        rag.build_or_load(source)
        _run_test_suite(rag, label, test_cfg["queries"])

    # ── DB 재로드 검증 ──
    md_tests = [t for t in filtered_tests if t["type"] == "markdown"]
    if md_tests:
        print(f"\n{'='*60}")
        print("[재로드 검증] 기존 DB를 디스크에서 재로드 후 검색")
        print(f"{'='*60}")
        rag_reload = MarkdownRAG(db_store_path=md_tests[0]["db_path"])
        rag_reload.build_or_load(md_tests[0]["source"])
        _print_results(
            "재로드 검증",
            "Endurance Group 정의 및 용도",
            rag_reload.retrieve("Endurance Group 정의 및 용도", top_k=2),
        )

    print(f"\n✅ 전체 테스트 완료!")

except Exception as e:
    print(f"\n❌ 실행 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 팁:")
    print("  - Ollama 서버 실행 중이고 OLLAMA_EMBEDDING_MODEL이 .env에 설정되어 있어야 합니다.")
    print("  - Gemini 사용 시 .env에 GEMINI_API_KEY와 GEMINI_EMBEDDING_MODEL을 설정하세요.")
    print("  - CodeRAG는 tree-sitter 설치 시 정확도가 크게 향상됩니다:")
    print("    pip install tree-sitter tree-sitter-python tree-sitter-c tree-sitter-cpp tree-sitter-java")

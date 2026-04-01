import os
import re
import sys
import hashlib
import pickle
from typing import List

# 상위 폴더(Dev_Agent) 모듈 참조 보장을 위해 PYTHONPATH 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from langchain_core.documents import Document
from rag import MarkdownRAG, CodeRAG

# MCP 서버 초기화
# 향후 여기에 추가하고 싶은 도구를 @mcp.tool() 데코레이터와 함께 정의하면 자동으로 Claude에 연동됩니다.
mcp = FastMCP("DeepAssist_MCP_Server")

# ──────────────────────────────────────────────
# 내부 헬퍼 함수
# ──────────────────────────────────────────────

def _auto_db_path(source_path: str) -> str:
    """source_path의 기반 이름 + MD5 앞 8자리로 고유한 DB 경로를 자동 결정한다.

    동일 source_path는 항상 동일 DB 경로를 반환하므로 재빌드 없이 재사용된다.
    """
    abs_path = os.path.abspath(os.path.expanduser(source_path))
    basename = os.path.splitext(os.path.basename(abs_path))[0]
    hash8 = hashlib.md5(abs_path.encode()).hexdigest()[:8]
    return os.path.expanduser(f"~/.deepassist/knowledge/{basename}_{hash8}")


def _rrf_merge(results_per_db: List[List[Document]], top_k: int, k: int = 60) -> List[Document]:
    """Reciprocal Rank Fusion으로 여러 DB의 검색 결과를 병합한다.

    각 결과의 순위(rank)를 기반으로 score = Σ 1/(k + rank + 1)을 계산한다.
    동일 청크가 여러 DB에서 높은 순위로 등장할수록 점수가 합산되어 우선 노출된다.
    """
    scores: dict = {}
    docs_map: dict = {}

    for results in results_per_db:
        for rank, doc in enumerate(results):
            # 청크 본문 앞 150자를 키로 사용 (동일 문서 중복 식별)
            key = doc.page_content[:150]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            docs_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [docs_map[key] for key in sorted_keys[:top_k]]


def _detect_rag_type(db_path: str) -> str:
    """DB 경로 내 인덱스 파일을 확인하여 RAG 유형(markdown/code)을 자동 감지한다.

    symbol_index.pkl이 존재하면 CodeRAG, term_index.pkl이 존재하면 MarkdownRAG로 판단.
    """
    if os.path.exists(os.path.join(db_path, "symbol_index.pkl")):
        return "code"
    return "markdown"


def _detect_source_type(source_path: str) -> str:
    """소스 경로의 파일 확장자 분포로 RAG 유형(markdown/code)을 자동 결정한다.

    코드 파일과 마크다운 파일이 모두 있으면 코드 파일 비율에 따라 결정.
    """
    code_exts = {'.py', '.c', '.cpp', '.h', '.hpp', '.cc', '.java', '.js', '.ts'}
    md_exts = {'.md'}

    source_path = os.path.abspath(os.path.expanduser(source_path))
    if os.path.isfile(source_path):
        ext = os.path.splitext(source_path)[1].lower()
        return "code" if ext in code_exts else "markdown"

    code_count = 0
    md_count = 0
    for root, dirs, filenames in os.walk(source_path):
        dirs[:] = [d for d in dirs if d not in {'.venv', 'venv', 'node_modules', '__pycache__', '.git', 'build', 'dist'}]
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in code_exts:
                code_count += 1
            elif ext in md_exts:
                md_count += 1

    if code_count > 0 and md_count == 0:
        return "code"
    elif md_count > 0 and code_count == 0:
        return "markdown"
    elif code_count > md_count:
        return "code"
    return "markdown"


def _load_and_search(db_path: str, query: str, top_k: int) -> List[Document]:
    """단일 DB를 로드하고 검색 결과를 반환한다. DB가 없으면 빈 리스트 반환.

    DB 유형(MarkdownRAG/CodeRAG)을 자동 감지하여 적절한 클래스로 로드한다.
    """
    db_path = os.path.abspath(os.path.expanduser(db_path))
    rag_type = _detect_rag_type(db_path)

    if rag_type == "code":
        rag = CodeRAG(db_store_path=db_path)
    else:
        rag = MarkdownRAG(db_store_path=db_path)

    if not rag.is_db_exists():
        return []
    rag.load_db()
    return rag.retrieve(query, top_k=top_k)


def _format_results(docs: List[Document]) -> str:
    """검색 결과 Document 리스트를 에이전트가 읽기 쉬운 문자열로 포맷팅한다.

    마크다운 RAG 결과와 코드 RAG 결과를 자동 구분하여 적절한 형식으로 출력.
    """
    output = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        source = meta.get("source", "Unknown")
        chunk_type = meta.get("chunk_type", "")

        if chunk_type in ("file_summary", "function", "class", "subchunk", "declarations"):
            # ── 코드 RAG 결과 포맷 ──
            language = meta.get("language", "")
            hierarchy = meta.get("hierarchy", "")
            signature = meta.get("signature", "")
            line_range = meta.get("line_range", "")
            req_ids = meta.get("requirement_ids", [])

            project = meta.get("project", "")
            header_parts = [f"[결과 {i+1}]"]
            if project:
                header_parts.append(f"프로젝트: {project}")
            header_parts.append(f"파일: {source}")
            if language:
                header_parts.append(f"언어: {language}")
            if chunk_type:
                header_parts.append(f"유형: {chunk_type}")
            if line_range:
                header_parts.append(f"라인: {line_range[0]}-{line_range[1]}")
            if hierarchy:
                header_parts.append(f"계층: {hierarchy}")
            if req_ids:
                header_parts.append(f"Req IDs: {', '.join(req_ids[:10])}")
            header = " | ".join(header_parts)
            if signature and chunk_type != "file_summary":
                header += f"\n  시그니처: {signature}"
        else:
            # ── 마크다운 RAG 결과 포맷 ──
            doc_name = meta.get("doc_name", "")
            section = meta.get("section", meta.get("Header 2", "N/A"))
            page = meta.get("page", "")
            doc_info = f"문서: {doc_name} | " if doc_name else ""
            page_info = f" | 페이지: {page}" if page else ""
            header = f"[결과 {i+1}] {doc_info}파일: {source} | 섹션: {section}{page_info}"

        output.append(f"{header}\n{doc.page_content.strip()}")
    return ("\n" + "-"*50 + "\n").join(output)


# ──────────────────────────────────────────────
# MCP 도구 정의
# ──────────────────────────────────────────────

@mcp.tool()
def build_knowledge_db(source_path: str, db_path: str = "", force_rebuild: bool = False) -> str:
    """
    지정된 파일 또는 폴더를 청킹·임베딩하여 Vector DB(FAISS + BM25)를 구축하고 디스크에 저장합니다.

    - source_path : 인덱싱할 파일/폴더 경로 (마크다운 .md 또는 소스코드 .py/.c/.cpp/.java 등)
                    소스 경로의 파일 확장자를 자동 분석하여 MarkdownRAG 또는 CodeRAG를 선택합니다.
    - db_path     : DB를 저장할 폴더 경로. 비워두면 ~/.deepassist/knowledge/<이름>_<hash>/ 에 자동 저장
    - force_rebuild: True이면 기존 DB가 있어도 재구축 (문서/코드 업데이트 시 사용)

    반환값: 구축(또는 재사용)된 DB의 절대 경로 문자열
    """
    resolved_db_path = db_path.strip() if db_path.strip() else _auto_db_path(source_path)
    resolved_db_path = os.path.abspath(os.path.expanduser(resolved_db_path))

    try:
        # 소스 유형 자동 감지 (마크다운 vs 코드)
        src_type = _detect_source_type(source_path)

        if src_type == "code":
            rag = CodeRAG(db_store_path=resolved_db_path)
        else:
            rag = MarkdownRAG(db_store_path=resolved_db_path)

        if rag.is_db_exists() and not force_rebuild:
            return f"✅ 기존 DB 재사용 ({src_type}): {resolved_db_path}\n(재구축하려면 force_rebuild=True 로 호출하세요)"

        rag._build_from_source(source_path)
        return f"✅ DB 구축 완료 ({src_type}): {resolved_db_path}"

    except Exception as e:
        return f"❌ DB 구축 중 오류가 발생했습니다: {e}"


def _get_knowledge_base_dir() -> str:
    """knowledge DB 기본 저장 디렉토리를 반환한다."""
    return os.path.expanduser("~/.deepassist/knowledge")


def _keyword_in_query(kw: str, query_lower: str) -> bool:
    """키워드가 쿼리에 독립 토큰으로 포함되는지 확인한다.

    단순 서브스트링 매칭(Python in 연산자)은 "uni"가 "uni92k" 안에서 매칭되는
    오탐을 발생시킨다. 이 함수는 키워드 앞뒤의 ASCII 영숫자 경계를 검사하여
    독립적으로 등장하는 경우에만 True를 반환한다.

    한글은 ASCII가 아니므로 경계로 인정: "pynvme로" → "pynvme" 매칭 OK
    ASCII 영숫자는 경계가 아님: "uni92k" 안의 "uni" → 뒤에 "9"(ASCII) → 매칭 거부

    예시:
    - "uni92k" in "uni92k 코드 만들어줘" → True  (뒤: 공백)
    - "uni"   in "uni92k 코드 만들어줘" → False (뒤: "9", ASCII 영숫자)
    - "uni"   in "uni 프레임워크"       → True  (뒤: 공백)
    - "nvme"  in "nvme에서 telemetry"   → True  (뒤: "에", 비ASCII)
    - "nvme"  in "pynvme로 에러"        → False (앞: "y", ASCII 영숫자)
    """
    start = 0
    while True:
        idx = query_lower.find(kw, start)
        if idx == -1:
            return False

        # 앞 글자 경계 검사: ASCII 영숫자가 아니어야 함
        if idx > 0:
            ch = query_lower[idx - 1]
            if ch.isascii() and ch.isalnum():
                start = idx + 1
                continue

        # 뒷 글자 경계 검사
        end = idx + len(kw)
        if end < len(query_lower):
            ch = query_lower[end]
            if ch.isascii() and ch.isalnum():
                start = idx + 1
                continue

        return True


def _match_dbs_by_query(query: str, all_dbs: List[dict]) -> List[dict]:
    """쿼리 텍스트에서 DB명/폴더명과 매칭되는 DB를 선별한다.

    매칭 전략:
    1. 각 DB의 name에서 키워드를 3단계로 생성:
       - 영숫자 혼합 토큰: ["uni92k"] — 정확한 매칭 (uni92K vs uni93K 구분)
       - 전체 이름 원형: "uni92k" — 공백/특수문자 제거
       - 영문 전용 토큰: ["uni"] — 폴백 매칭
    2. 독립 토큰 매칭: 키워드가 쿼리에서 독립적으로 등장하는지 검사
       (서브스트링 오매칭 방지: "uni"가 "uni92k" 안에서 매칭되지 않음)
    3. 매칭된 DB가 있으면 해당 DB만 반환, 없으면 빈 리스트 → 전체 검색

    예시:
    - "uni92K 코드 만들어줘" → "uni92k" 매칭 → uni92K DB만 선택 (uni93K 제외)
    - "uni 프레임워크 공통" → "uni" 매칭 → uni92K, uni93K 모두 선택
    - "nvme에서 telemetry" → "nvme" 매칭 → NVMe 관련 DB 선택
    - "telemetry 로그 구조" → 매칭 없음 → 전체 검색
    """
    query_lower = query.lower()

    # 1단계: 각 DB별 키워드 생성 (긴 것 → 짧은 것 순으로 정렬)
    db_keywords: List[tuple] = []  # [(db, [keywords_longest_first]), ...]
    for db in all_dbs:
        db_name = db.get("name", "")
        keywords: List[str] = []

        # 전체 이름 (공백/특수문자 제거한 원형)
        full_name = re.sub(r'[^a-zA-Z0-9]', '', db_name).lower()
        if len(full_name) >= 2:
            keywords.append(full_name)

        # 영숫자 혼합 토큰 (uni92K → uni92k, DeviceA → devicea)
        alnum_tokens = re.findall(r'[a-zA-Z0-9]{2,}', db_name)
        keywords.extend([t.lower() for t in alnum_tokens])

        # 영문 전용 토큰 (폴백용, nvme, ocp 등 단순 키워드)
        alpha_tokens = re.findall(r'[a-zA-Z]{2,}', db_name)
        keywords.extend([t.lower() for t in alpha_tokens])

        # 폴더명에서도 추출 (자동 생성된 DB명 보완)
        folder_name = os.path.basename(db.get("path", ""))
        # 해시 접미사 제거: "uni92K_abc12345" → "uni92K"
        folder_base = folder_name.rsplit("_", 1)[0] if "_" in folder_name else folder_name
        folder_alnum = re.findall(r'[a-zA-Z0-9]{2,}', folder_base)
        keywords.extend([t.lower() for t in folder_alnum])

        # 중복 제거 + 긴 것부터 정렬 (긴 키워드 우선 매칭)
        keywords = sorted(set(keywords), key=len, reverse=True)
        db_keywords.append((db, keywords))

    # 2단계: 독립 토큰 매칭 (서브스트링 오매칭 방지)
    matched: List[dict] = []
    for db, keywords in db_keywords:
        for kw in keywords:
            if _keyword_in_query(kw, query_lower):
                matched.append(db)
                break

    return matched


def _list_all_dbs() -> List[dict]:
    """구축된 모든 knowledge DB 목록을 반환한다.

    각 DB 디렉토리에서 메타 정보(doc_meta.pkl 또는 project_meta.pkl)를 읽어
    유형, 이름, 경로를 포함한 딕셔너리 리스트로 반환한다.
    """
    base_dir = _get_knowledge_base_dir()
    if not os.path.exists(base_dir):
        return []

    dbs = []
    for entry in sorted(os.listdir(base_dir)):
        db_path = os.path.join(base_dir, entry)
        if not os.path.isdir(db_path):
            continue
        # FAISS 인덱스가 없으면 유효한 DB가 아님
        if not os.path.exists(os.path.join(db_path, "faiss_index")):
            continue

        db_info = {"path": db_path, "type": "unknown", "name": entry}

        # CodeRAG DB인지 확인
        project_meta_path = os.path.join(db_path, "project_meta.pkl")
        if os.path.exists(project_meta_path):
            try:
                with open(project_meta_path, "rb") as f:
                    meta = pickle.load(f)
                db_info["type"] = "code"
                db_info["name"] = meta.get("project_name", entry)
                db_info["project_root"] = meta.get("project_root", "")
            except Exception:
                db_info["type"] = "code"

        # MarkdownRAG DB인지 확인
        doc_meta_path = os.path.join(db_path, "doc_meta.pkl")
        if os.path.exists(doc_meta_path):
            try:
                with open(doc_meta_path, "rb") as f:
                    meta = pickle.load(f)
                db_info["type"] = "markdown"
                db_info["name"] = meta.get("doc_name", entry)
            except Exception:
                db_info["type"] = "markdown"

        dbs.append(db_info)

    return dbs


@mcp.tool()
def list_knowledge_dbs() -> str:
    """
    구축된 모든 Knowledge DB 목록을 반환합니다.

    각 DB의 유형(markdown/code), 이름, 경로를 표시합니다.
    search_knowledge 호출 시 db_path 파라미터에 사용할 경로를 확인할 수 있습니다.
    여러 DB를 동시에 검색하려면 쉼표로 구분하여 db_path에 전달하세요.
    """
    dbs = _list_all_dbs()
    if not dbs:
        return "❌ 구축된 Knowledge DB가 없습니다.\nbuild_knowledge_db()로 먼저 DB를 구축하세요."

    lines = [f"📚 구축된 Knowledge DB 목록 ({len(dbs)}개):", ""]
    for i, db in enumerate(dbs, 1):
        db_type = db['type'].upper()
        name = db['name']
        path = db['path']
        extra = ""
        if db.get("project_root"):
            extra = f"\n     소스: {db['project_root']}"
        lines.append(f"  {i}. [{db_type}] {name}")
        lines.append(f"     경로: {path}{extra}")

    lines.append("")
    lines.append("💡 검색 시 db_path에 위 경로를 지정하세요.")
    lines.append("   여러 DB 동시 검색: db_path=\"경로1,경로2\"")
    return "\n".join(lines)


@mcp.tool()
def search_knowledge(query: str, db_path: str = "", top_k: int = 4) -> str:
    """
    Vector DB(FAISS + BM25 하이브리드)에서 기술 문서를 검색합니다.

    - query   : 검색할 키워드, 기술 개념, 오류 메시지 등
    - db_path : 검색할 DB 경로.
                  • 단일 DB: "/path/to/db"
                  • 여러 DB 동시 검색(Fan-out): "/path/db1,/path/db2,/path/db3"
                    (쉼표 구분, 결과는 RRF 알고리즘으로 통합 순위 결정)
                  • 비워두면 쿼리에서 DB명을 자동 감지하여 관련 DB만 검색
                    (매칭 없으면 전체 DB Fan-out 검색)
                  구축된 DB 목록은 list_knowledge_dbs()로 확인할 수 있습니다.
    - top_k   : 반환할 결과 수 (기본 4, 너무 많으면 컨텍스트 초과 위험)

    반환값: 검색된 청크 목록 (파일명, 섹션, 페이지, 본문 포함)
    """
    # db_path 결정:
    #   1) 명시적 지정 → 그대로 사용
    #   2) 미지정 → 쿼리에서 DB명 자동 매칭 → 관련 DB만 선택
    #   3) 매칭 없음 → 전체 DB Fan-out
    if db_path.strip():
        db_paths = [p.strip() for p in db_path.split(",") if p.strip()]
    else:
        all_dbs = _list_all_dbs()
        if not all_dbs:
            return ("❌ 구축된 Knowledge DB가 없습니다.\n"
                    "build_knowledge_db(source_path)로 먼저 DB를 구축하세요.")

        # 쿼리에서 DB명 키워드 자동 매칭
        matched_dbs = _match_dbs_by_query(query, all_dbs)
        if matched_dbs:
            db_paths = [db["path"] for db in matched_dbs]
        else:
            # 매칭 없음 → 전체 DB Fan-out
            db_paths = [db["path"] for db in all_dbs]

    try:
        if len(db_paths) == 1:
            # ── 단일 DB 검색 ──────────────────────────────────
            docs = _load_and_search(db_paths[0], query, top_k)
            if not docs:
                return f"❌ DB에서 관련 문서를 찾을 수 없습니다. (DB 경로: {db_paths[0]})\nbuild_knowledge_db()로 먼저 DB를 구축했는지 확인하세요."
            return _format_results(docs)

        else:
            # ── Fan-out: 여러 DB 병렬 검색 후 RRF 병합 ────────
            results_per_db: List[List[Document]] = []
            missing = []

            for path in db_paths:
                results = _load_and_search(path, query, top_k)
                if results:
                    results_per_db.append(results)
                else:
                    missing.append(path)

            if not results_per_db:
                return f"❌ 지정된 DB({', '.join(db_paths)}) 모두에서 결과를 찾지 못했습니다."

            merged = _rrf_merge(results_per_db, top_k)
            header = f"🔍 {len(db_paths)}개 DB Fan-out 검색 | RRF 통합 결과 (top {top_k})"
            if missing:
                header += f"\n⚠️ DB 없음(건너뜀): {', '.join(missing)}"

            return header + "\n" + "="*50 + "\n" + _format_results(merged)

    except Exception as e:
        return f"❌ 문서 검색 중 오류가 발생했습니다: {e}"

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

@mcp.tool()
def search_web_and_scrape(query: str, max_results: int = 2) -> str:
    """
    DuckDuckGo를 통해 웹을 검색하고, 가장 상위 페이지의 본문을 추출(Scraping)하여 반환합니다.
    크롤링이 차단된 사이트(예: 403 Forbidden)는 자동으로 건너뛰고 다음 링크를 시도합니다.

    - query: 검색어
    - max_results: 최대로 긁어올 성공적인 페이지 수 (추천: 1~2)
    """
    try:
        results = DDGS().text(query, max_results=max_results + 3) # 차단 대비 여유있게 가져옴
        if not results:
            return "❌ 검색 결과가 없습니다."

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        scraped_texts = []
        success_count = 0

        for res in results:
            if success_count >= max_results:
                break

            href = res.get("href")
            title = res.get("title", '제목 없음')
            # 딕셔너리 리스트 반환 방식에 따른 호환성
            if not href:
                # DDG 버전에 따라 url이 키일 수도 있음
                href = res.get("url")
            if not href:
                continue

            try:
                resp = requests.get(href, headers=headers, timeout=8)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    # 헤더, 푸터, 스크립트 등 불필요한 태그 파괴
                    for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'iframe', 'svg']):
                        tag.extract()
                    # 텍스트 추출 (연속된 공백 제거)
                    text = ' '.join(soup.get_text(separator=' ', strip=True).split())

                    # 텍스트가 너무 길면 자르기 (LLM 컨텍스트 오버플로우 방지 5000자)
                    if len(text) > 5000:
                        text = text[:5000] + "\n...(중략)..."

                    snippet = res.get('body', '')
                    scraped_texts.append(f"### [웹 검색 결과 {success_count+1}] {title}\n🔗 출처: {href}\n📝 스니펫: {snippet}\n\n{text}")
                    success_count += 1
                else:
                    # 403 Forbidden 등 에러 발생 시 다음 링크 검색
                    continue
            except Exception:
                # 타임아웃, 접속 에러 시 다음 링크 검색
                continue

        if not scraped_texts:
            return "❌ 상위 사이트들이 스크래핑을 차단(403)했거나 접속할 수 없습니다. 검색어를 바꿔서 다시 시도해주세요."

        return "\n\n" + "="*50 + "\n\n" + ("\n\n" + "="*50 + "\n\n").join(scraped_texts)

    except Exception as e:
        return f"❌ 웹 검색/스크래핑 중 오류가 발생했습니다: {str(e)}"

if __name__ == "__main__":
    # Claude-Code SDK는 stdio 파이프 방식으로 MCP 서버와 통신합니다.
    mcp.run(transport='stdio')

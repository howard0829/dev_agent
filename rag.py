"""rag.py — FAISS + BM25 하이브리드 RAG 모듈 (마크다운 + 소스코드 통합)

마크다운 기술문서(MarkdownRAG)와 소스코드(CodeRAG)를 위한 통합 RAG 시스템.
청킹, 임베딩, 벡터 DB 구축, 하이브리드 검색을 모두 이 모듈에서 수행한다.

아키텍처:
  BaseRAG          — 공통 인프라 (임베딩, FAISS/BM25, 앙상블 검색)
  ├── MarkdownRAG  — 마크다운 기술문서 전용 (헤더 경계 청킹, 용어 인덱스)
  └── CodeRAG      — 소스코드 전용 (tree-sitter AST 청킹, 심볼/파일/함수 인덱스)
"""

import os
import hashlib
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# ── 임베딩 프로바이더 ──────────────────────────────────────────
from langchain_ollama import OllamaEmbeddings
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:
    GoogleGenerativeAIEmbeddings = None

# ── tree-sitter (선택적 의존성 — 미설치 시 regex 폴백 사용) ───
# tree-sitter는 소스코드를 AST(추상 구문 트리)로 파싱하여
# 함수/클래스 경계를 정확하게 감지하는 데 사용된다.
# 미설치 시에도 CodeRAG는 regex 기반 폴백으로 동작한다.
_TREE_SITTER_AVAILABLE = False
try:
    from tree_sitter import Language as TSLanguage, Parser as TSParser
    _TREE_SITTER_AVAILABLE = True
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════
# 공통 상수 및 정규식 패턴
# ══════════════════════════════════════════════════════════════

# ── 마크다운 전용 패턴 ─────────────────────────────────────────
# 마크다운 헤더 패턴 (# ~ ######)
_HEADER_RE = re.compile(r'^(#{1,6})\s+(.*)')
# 페이지 마커 패턴 (<!-- page: N -->)
_PAGE_RE = re.compile(r'<!--\s*page:\s*(\d+)\s*-->')
# 테이블 행 패턴 (|로 시작하는 라인, 구분선 |---|---| 제외)
_TABLE_ROW_RE = re.compile(r'^\|(?![-|:\s]+\|$)')
_TABLE_SEP_RE = re.compile(r'^\|[-|:\s]+\|$')

# ── 공통 패턴 (마크다운 + 코드 모두 사용) ─────────────────────
# Requirement ID 패턴 (TEL-6, SEC-3, FWUP-15 등)
# 접두사 2글자 이상의 대문자 + 하이픈 + 숫자로 구성
# UTF-8, AES-256 같은 일반 기술 용어 오탐 방지를 위해 제외 목록 적용
_REQ_ID_RE = re.compile(r'\b([A-Z]{2,}[A-Z0-9]*-\d+)\b')
_REQ_ID_EXCLUDE = {'UTF-8', 'AES-128', 'AES-256', 'SHA-256', 'SHA-384', 'SHA-512',
                    'FIPS-140', 'IEEE-1667', 'SP-800', 'X-509',
                    'JESD218B-02', 'JESD218A-01'}
# 약어 정의 패턴: "Full Name (ABBR)" — 기술문서에서 용어 정의 시 사용
_ABBR_DEF_RE = re.compile(r'([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)\s*\(([A-Z][A-Z0-9]{1,})\)')

# ── 코드 전용 패턴 ────────────────────────────────────────────
# 코드 내 언더스코어 기반 Requirement ID 패턴
# 함수명 등에서 TEL_2, SEC_5 형태로 사용되는 것을 TEL-2, SEC-5로 정규화
_CODE_REQ_PATTERN = re.compile(r'\b([A-Z]{2,}[A-Z0-9]*)_(\d+)\b')

# ── tree-sitter 언어 레지스트리 ────────────────────────────────
# 각 언어별로 tree-sitter 그래머 모듈명, 함수/클래스/네임스페이스 노드 타입,
# import 노드 타입, 데코레이터 인식 여부를 정의한다.
# 새 언어를 추가하려면 이 딕셔너리에 한 줄만 추가하면 된다.
_LANGUAGE_CONFIG = {
    ".py": {
        "name": "python",
        "grammar_module": "tree_sitter_python",
        # Python은 decorated_definition이 데코레이터+함수를 감싸므로 우선 수집
        # @pytest.mark.parametrize, @test_case("TEL-2") 등 데코레이터가 함수와 함께 청킹됨
        "func_types": ["function_definition", "decorated_definition"],
        "class_types": ["class_definition"],
        "namespace_types": [],
        "import_types": ["import_statement", "import_from_statement"],
        "decorator_aware": True,  # decorated_definition 내부 function은 중복 수집 방지
        "body_types": ["block"],  # 함수 본문 노드 타입 (시그니처 추출 시 사용)
    },
    ".c": {
        "name": "c",
        "grammar_module": "tree_sitter_c",
        "func_types": ["function_definition"],
        "class_types": ["struct_specifier", "enum_specifier"],
        "namespace_types": [],
        "import_types": ["preproc_include"],
        "decorator_aware": False,
        "body_types": ["compound_statement"],
    },
    ".h": {
        "name": "c",
        "grammar_module": "tree_sitter_c",
        "func_types": ["function_definition"],
        "class_types": ["struct_specifier", "enum_specifier"],
        "namespace_types": [],
        "import_types": ["preproc_include"],
        "decorator_aware": False,
        "body_types": ["compound_statement"],
    },
    ".cpp": {
        "name": "cpp",
        "grammar_module": "tree_sitter_cpp",
        "func_types": ["function_definition"],
        "class_types": ["class_specifier", "struct_specifier"],
        "namespace_types": ["namespace_definition"],
        "import_types": ["preproc_include"],
        "decorator_aware": False,
        "body_types": ["compound_statement", "field_declaration_list"],
    },
    ".hpp": {
        "name": "cpp",
        "grammar_module": "tree_sitter_cpp",
        "func_types": ["function_definition"],
        "class_types": ["class_specifier", "struct_specifier"],
        "namespace_types": ["namespace_definition"],
        "import_types": ["preproc_include"],
        "decorator_aware": False,
        "body_types": ["compound_statement", "field_declaration_list"],
    },
    ".cc": {
        "name": "cpp",
        "grammar_module": "tree_sitter_cpp",
        "func_types": ["function_definition"],
        "class_types": ["class_specifier", "struct_specifier"],
        "namespace_types": ["namespace_definition"],
        "import_types": ["preproc_include"],
        "decorator_aware": False,
        "body_types": ["compound_statement", "field_declaration_list"],
    },
    ".java": {
        "name": "java",
        "grammar_module": "tree_sitter_java",
        # Java 어노테이션(@Override 등)은 method_declaration 노드 안에 포함되어 별도 처리 불필요
        "func_types": ["method_declaration", "constructor_declaration"],
        "class_types": ["class_declaration", "interface_declaration", "enum_declaration"],
        "namespace_types": [],
        "import_types": ["import_declaration"],
        "decorator_aware": False,
        "body_types": ["block"],
    },
}

# ── tree-sitter 그래머 캐시 ────────────────────────────────────
# {확장자: TSLanguage} — 설치된 그래머만 로드하여 캐시
_TS_LANGUAGES: Dict[str, 'TSLanguage'] = {}
# {확장자: TSParser} — 파서 인스턴스 캐시 (재사용으로 성능 향상)
_TS_PARSERS: Dict[str, 'TSParser'] = {}


def _init_ts_grammars() -> None:
    """설치된 tree-sitter 언어 그래머를 감지하여 로드한다.

    각 언어 그래머 패키지(tree_sitter_python, tree_sitter_c 등)를
    독립적으로 try/except 하여, 일부만 설치되어 있어도 동작한다.
    """
    if not _TREE_SITTER_AVAILABLE:
        return

    loaded = set()  # 이미 로드한 모듈 (같은 모듈을 여러 확장자가 공유)
    for ext, config in _LANGUAGE_CONFIG.items():
        module_name = config["grammar_module"]
        if module_name in loaded:
            # 같은 그래머를 공유하는 확장자 (.c와 .h, .cpp와 .hpp 등)
            # 이미 로드된 언어 객체를 공유
            for other_ext, other_config in _LANGUAGE_CONFIG.items():
                if other_config["grammar_module"] == module_name and other_ext in _TS_LANGUAGES:
                    _TS_LANGUAGES[ext] = _TS_LANGUAGES[other_ext]
                    break
            continue
        try:
            mod = __import__(module_name)
            lang = TSLanguage(mod.language())
            _TS_LANGUAGES[ext] = lang
            loaded.add(module_name)
        except (ImportError, AttributeError, OSError):
            # 해당 언어 그래머 미설치 → 이 확장자는 regex 폴백 사용
            pass


# 모듈 로드 시 한 번만 실행
_init_ts_grammars()


# ══════════════════════════════════════════════════════════════
# 마크다운 청킹 유틸리티
# ══════════════════════════════════════════════════════════════

def bm25_preprocessor(text: str) -> str:
    """마크다운 BM25 전처리: 특수문자 제거, Requirement ID 하이픈 보존 후 소문자화.

    "TEL-6" → "tel-6" (하이픈 보존, 단일 토큰 유지)
    "NVMe-oF" → "nvme of" (일반 하이픈은 공백 치환)
    """
    # 1단계: Requirement ID 패턴(대문자-숫자)의 하이픈을 임시 플레이스홀더로 치환
    processed = _REQ_ID_RE.sub(
        lambda m: m.group(0).replace('-', '\x00') if m.group(0) not in _REQ_ID_EXCLUDE else m.group(0),
        text,
    )
    # 2단계: 나머지 특수문자 제거 (영문/숫자/한글/공백만 보존)
    processed = re.sub(r'[^a-zA-Z0-9가-힣\s\x00]', ' ', processed)
    # 3단계: 플레이스홀더를 하이픈으로 복원
    processed = processed.replace('\x00', '-')
    return processed.lower()


def _extract_chunk_context(page_content: str) -> str:
    """원본 청크에서 2차 분할 시 서브 청크에 주입할 구조적 문맥을 추출.

    추출 대상 (등장 순서대로):
    - [source | 섹션: ...] 접두사
    - 마크다운 헤더 (# ~ ######)
    - 테이블 헤더행 + 구분선 (첫 번째 테이블만)
    """
    context_lines: List[str] = []
    lines = page_content.split('\n')
    table_header_found = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        # [source | 섹션: ...] 접두사
        if stripped.startswith('[') and '섹션:' in stripped:
            context_lines.append(line)
            continue
        # 마크다운 헤더
        if _HEADER_RE.match(stripped):
            context_lines.append(line)
            continue
        # 테이블 헤더: |...| 행 바로 뒤에 |---|---| 구분선이 오는 패턴 (첫 번째만)
        if (not table_header_found
                and _TABLE_ROW_RE.match(stripped)
                and i + 1 < len(lines)
                and _TABLE_SEP_RE.match(lines[i + 1].strip())):
            context_lines.append(line)
            context_lines.append(lines[i + 1])
            table_header_found = True

    return '\n'.join(context_lines)


def _find_content_start(lines: List[str], min_body_chars: int = 100) -> int:
    """목차(TOC) 등 프론트매터를 건너뛰고 실제 본문이 시작되는 라인 인덱스를 반환.

    헤더 뒤에 실질적 본문(min_body_chars 이상)이 있는 첫 헤더를 콘텐츠 시작점으로 판단.
    목차는 헤더가 빽빽하게 나열되고 본문이 거의 없는 구간이므로 자연스럽게 스킵됨.
    """
    for i, line in enumerate(lines):
        if not _HEADER_RE.match(line.strip()):
            continue
        body_len = 0
        for j in range(i + 1, len(lines)):
            next_stripped = lines[j].strip()
            if _HEADER_RE.match(next_stripped):
                break
            if next_stripped == '---' or _PAGE_RE.search(next_stripped):
                continue
            body_len += len(next_stripped)
        if body_len >= min_body_chars:
            return i
    return 0


def _split_md_by_header_boundary(
    content: str,
    source: str,
    min_chunk_size: int = 1000,
    max_chunk_size: int = 3000,
) -> List[Document]:
    """마크다운을 헤더 경계 기준으로 분할.

    동작 방식:
    - 프론트매터 스킵: 목차(TOC), 표지 등 헤더만 나열된 구간을 자동 감지하여 건너뜀
      (헤더 뒤 본문이 100자 이상인 첫 헤더부터 청킹 시작)
    - 라인을 순차적으로 누적하며 현재 청크 크기(문자 수)를 추적
    - 현재 청크가 min_chunk_size 이상 누적된 상태에서 헤더(# ~ ######)를 만나면
      그 직전에서 청크를 확정하고 헤더부터 새 청크 시작
    - min_chunk_size 미만일 때는 헤더를 만나도 계속 누적
    - <!-- page: N --> 마커에서 현재 페이지 번호를 추적 (청크에 미포함)
    - 수평선(---) 스킵: 페이지 구분용 노이즈이므로 청크에 포함하지 않음
    - 각 청크 앞에 "[source | 섹션: H1 > H2 > ...]" 형태의 문맥 접두사 주입
    - max_chunk_size 초과 청크는 테이블 행 경계 > 단락 경계(\\n\\n) 순으로 2차 분할
      (2차 분할 시 구조적 문맥(접두사+마크다운 헤더+테이블 헤더행+구분선) 자동 주입)
    """
    raw_chunks: List[Document] = []
    current_lines: List[str] = []
    current_len: int = 0
    current_page: int = 1
    header_stack: dict = {}  # {레벨(int): 헤더 텍스트}

    # 테이블 상태 추적
    in_table: bool = False
    table_header_lines: List[str] = []  # 테이블 헤더행 + 구분선 (페이지 경계 중복 감지용)

    def flush() -> None:
        nonlocal current_lines, current_len
        text = '\n'.join(current_lines).strip()
        text = re.sub(r'\n{3,}', '\n\n', text)
        if not text:
            current_lines = []
            current_len = 0
            return
        path_parts = [header_stack[lvl] for lvl in sorted(header_stack.keys())]
        section_path = " > ".join(path_parts)
        prefix = f"[{source} | 섹션: {section_path}]\n" if section_path else f"[{source}]\n"
        req_ids = sorted(set(_REQ_ID_RE.findall(text)) - _REQ_ID_EXCLUDE)
        abbr_matches = _ABBR_DEF_RE.findall(text)
        abbreviations = {abbr: full_name for full_name, abbr in abbr_matches}
        raw_chunks.append(Document(
            page_content=prefix + text,
            metadata={
                "source": source,
                "section": section_path,
                "page": current_page,
                "requirement_ids": req_ids,
                "abbreviations": abbreviations,
            },
        ))
        current_lines = []
        current_len = 0

    def update_header_stack(level: int, header_text: str) -> None:
        header_stack[level] = header_text
        for k in list(header_stack.keys()):
            if k > level:
                del header_stack[k]

    all_lines = content.split('\n')
    content_start = _find_content_start(all_lines)

    for line in all_lines[content_start:]:
        stripped = line.strip()

        page_m = _PAGE_RE.search(line)
        if page_m:
            current_page = int(page_m.group(1))
            continue

        if stripped == '---':
            continue

        if in_table and not stripped:
            continue

        # ── 테이블 구분선 (|---|---|) ──
        if _TABLE_SEP_RE.match(stripped):
            new_col_count = stripped.count('|')
            if in_table and table_header_lines:
                existing_seps = [ln for ln in table_header_lines if _TABLE_SEP_RE.match(ln.strip())]
                existing_col_count = existing_seps[0].count('|') if existing_seps else 0
                if new_col_count == existing_col_count:
                    if current_lines and _TABLE_ROW_RE.match(current_lines[-1].strip()):
                        dup_line = current_lines.pop()
                        current_len -= len(dup_line) + 1
                    continue
                else:
                    current_lines.append(line)
                    current_len += len(line) + 1
                    continue

            in_table = True
            table_header_lines.clear()
            if current_lines and _TABLE_ROW_RE.match(current_lines[-1].strip()):
                table_header_lines.append(current_lines[-1])
            table_header_lines.append(line)
            current_lines.append(line)
            current_len += len(line) + 1
            continue

        # ── 테이블 데이터 행 (|...|) ──
        if _TABLE_ROW_RE.match(stripped):
            if not in_table:
                current_lines.append(line)
                current_len += len(line) + 1
                continue
            current_lines.append(line)
            current_len += len(line) + 1
            continue

        if in_table and stripped and not stripped.startswith('|'):
            in_table = False
            table_header_lines.clear()

        # ── 마크다운 헤더 (#~######) ──
        header_m = _HEADER_RE.match(line)
        if header_m and current_len >= min_chunk_size:
            flush()
            in_table = False
            table_header_lines.clear()
            level = len(header_m.group(1))
            update_header_stack(level, header_m.group(2).strip())
            current_lines = [line]
            current_len = len(line) + 1
        else:
            if header_m:
                level = len(header_m.group(1))
                update_header_stack(level, header_m.group(2).strip())
            current_lines.append(line)
            current_len += len(line) + 1

    flush()

    # 2차: max_chunk_size 초과 청크를 테이블 행 > 단락 경계로 재분할
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=200,
        separators=[r"\n(?=\|)", "\n\n", "\n", " ", ""],
        is_separator_regex=True,
    )
    result: List[Document] = []
    for doc in raw_chunks:
        if len(doc.page_content) <= max_chunk_size:
            result.append(doc)
        else:
            sub_docs = fallback_splitter.split_documents([doc])
            context = _extract_chunk_context(doc.page_content)
            for i, sub_doc in enumerate(sub_docs):
                if i > 0 and context:
                    sub_doc.page_content = context + '\n' + sub_doc.page_content
            result.extend(sub_docs)

    return result


# ══════════════════════════════════════════════════════════════
# 코드 청킹 유틸리티 (tree-sitter AST 기반 + regex 폴백)
# ══════════════════════════════════════════════════════════════

def code_bm25_preprocessor(text: str) -> str:
    """코드 BM25 전처리: CamelCase/snake_case 분리 + Req ID 하이픈 보존.

    코드 검색에 최적화된 전처리:
    - CamelCase를 구성 단어로 분리: "BlockManager" → "block manager blockmanager"
    - snake_case를 구성 단어로 분리: "allocate_block" → "allocate block allocate_block"
    - Requirement ID 하이픈 보존: "TEL-6" → "tel-6"
    - 코드 노이즈(중괄호, 세미콜론, 연산자 등) 제거
    """
    # 1단계: Requirement ID 하이픈 보존 (기존 마크다운 전처리와 동일)
    processed = _REQ_ID_RE.sub(
        lambda m: m.group(0).replace('-', '\x00') if m.group(0) not in _REQ_ID_EXCLUDE else m.group(0),
        text,
    )

    # 2단계: CamelCase 분리 (단어 사이에 공백 삽입)
    # "BlockManager" → "Block Manager", "parseHTTPResponse" → "parse HTTP Response"
    processed = re.sub(r'([a-z])([A-Z])', r'\1 \2', processed)
    processed = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', processed)

    # 3단계: 코드 노이즈 제거 (영문/숫자/한글/공백/언더스코어/플레이스홀더만 보존)
    processed = re.sub(r'[^a-zA-Z0-9가-힣\s_\x00]', ' ', processed)

    # 4단계: 언더스코어를 공백으로 치환 (원본도 별도 토큰으로 유지)
    # "allocate_block" → "allocate block" (BM25가 개별 단어로 검색 가능)
    processed = processed.replace('_', ' ')

    # 5단계: 플레이스홀더 복원 + 소문자화
    processed = processed.replace('\x00', '-')
    return processed.lower()


def _get_ts_parser(ext: str) -> Optional['TSParser']:
    """확장자에 해당하는 tree-sitter 파서를 반환. 미지원 시 None.

    파서는 확장자별로 캐시되어 동일 언어 파일을 반복 파싱해도 재생성하지 않는다.
    """
    if ext not in _TS_LANGUAGES:
        return None
    if ext not in _TS_PARSERS:
        parser = TSParser(_TS_LANGUAGES[ext])
        _TS_PARSERS[ext] = parser
    return _TS_PARSERS[ext]


def _get_node_name(node, source_bytes: bytes) -> Optional[str]:
    """AST 노드에서 이름(식별자)을 추출한다.

    함수명, 클래스명, 네임스페이스명 등을 가져올 때 사용.
    tree-sitter 노드의 자식 중 identifier/name/type_identifier를 찾는다.
    """
    name_types = {'identifier', 'name', 'type_identifier', 'field_identifier'}
    for child in node.children:
        if child.type in name_types:
            return source_bytes[child.start_byte:child.end_byte].decode('utf-8', errors='replace')
    # Python class_definition 등에서 name이 좀 더 깊이 있을 수 있음
    return None


def _get_signature(node, source_bytes: bytes, lang_config: dict) -> str:
    """함수/메서드 노드에서 시그니처(선언부)를 추출한다.

    body 노드(함수 본문) 이전까지의 텍스트를 시그니처로 사용.
    예: "void BlockManager::allocate_block(size_t size, int flags)"
    """
    # body 노드를 찾아 그 이전까지를 시그니처로 추출
    for child in node.children:
        if child.type in lang_config.get("body_types", []):
            sig = source_bytes[node.start_byte:child.start_byte].decode('utf-8', errors='replace').strip()
            # 여러 줄의 시그니처를 한 줄로 압축
            sig = ' '.join(sig.split())
            return sig
    # body가 없으면 첫 줄 사용
    text = source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
    return text.split('\n')[0].strip()


def _extract_hierarchy(node, source_bytes: bytes, lang_config: dict) -> str:
    """AST 노드의 부모 체인을 역추적하여 계층 경로를 추출한다.

    반환 예: "StorageEngine > BlockManager > allocate_block"

    namespace, class, function 등 의미 있는 스코프만 포함하며,
    translation_unit(파일 루트), block(본문) 등 구조적 노드는 건너뛴다.
    """
    parts: List[str] = []
    current = node.parent
    scope_types = set(
        lang_config.get("namespace_types", [])
        + lang_config.get("class_types", [])
        + lang_config.get("func_types", [])
    )

    while current is not None:
        if current.type in scope_types:
            name = _get_node_name(current, source_bytes)
            if name:
                parts.append(name)
        current = current.parent

    parts.reverse()
    return " > ".join(parts)


def _extract_imports(root_node, source_bytes: bytes, lang_config: dict) -> List[str]:
    """파일의 최상위 import/include 문을 추출한다.

    반환 예: ["#include <stdio.h>", "#include \"block_manager.h\""]
    """
    imports: List[str] = []
    import_types = set(lang_config.get("import_types", []))

    for child in root_node.children:
        if child.type in import_types:
            text = source_bytes[child.start_byte:child.end_byte].decode('utf-8', errors='replace').strip()
            imports.append(text)

    return imports


def _collect_target_nodes(
    node, lang_config: dict, source_bytes: bytes,
    results: Optional[List] = None, depth: int = 0,
    inside_decorated: bool = False,
) -> List[Tuple]:
    """AST를 재귀적으로 탐색하여 함수/클래스 노드를 수집한다.

    반환: [(node, depth, node_type), ...] 리스트
    - node: tree-sitter 노드
    - depth: 중첩 깊이 (파일 요약의 들여쓰기에 사용)
    - node_type: "function", "class", "decorated_function" 중 하나

    decorator_aware=True인 언어(Python)에서는:
    - decorated_definition을 만나면 "decorated_function"으로 수집
    - 그 내부의 function_definition은 중복 수집하지 않음
    """
    if results is None:
        results = []

    func_types = set(lang_config.get("func_types", []))
    class_types = set(lang_config.get("class_types", []))
    decorator_aware = lang_config.get("decorator_aware", False)

    if decorator_aware and node.type == "decorated_definition":
        # 데코레이터+함수를 하나의 단위로 수집
        results.append((node, depth, "decorated_function"))
        # 내부 class_definition은 별도 수집 (데코레이터 붙은 클래스)
        for child in node.children:
            if child.type in class_types:
                results.append((child, depth, "class"))
                # 클래스 내부 메서드도 수집
                for grandchild in child.children:
                    _collect_target_nodes(grandchild, lang_config, source_bytes, results, depth + 1)
        return results

    if node.type in func_types:
        if not inside_decorated:  # 데코레이터 내부가 아닐 때만 수집
            results.append((node, depth, "function"))

    elif node.type in class_types:
        results.append((node, depth, "class"))
        # 클래스 내부 메서드를 한 단계 깊이에서 수집
        for child in node.children:
            _collect_target_nodes(child, lang_config, source_bytes, results, depth + 1,
                                 inside_decorated=False)
        return results

    # 자식 노드 재귀 탐색
    is_in_decorated = inside_decorated or (decorator_aware and node.type == "decorated_definition")
    for child in node.children:
        _collect_target_nodes(child, lang_config, source_bytes, results, depth, is_in_decorated)

    return results


def _extract_req_ids_from_code(text: str) -> List[str]:
    """코드 텍스트에서 Requirement ID를 추출한다.

    두 가지 패턴을 인식:
    1. 하이픈 형태: TEL-2, SEC-5 (문자열 리터럴, 주석 등)
    2. 언더스코어 형태: TEL_2, SEC_5 (함수명, 변수명) → TEL-2, SEC-5로 정규화

    _REQ_ID_EXCLUDE에 포함된 일반 기술 용어(UTF-8, AES-256 등)는 제외.
    """
    upper_text = text.upper()

    # 하이픈 형태 Req ID
    req_ids = set(_REQ_ID_RE.findall(upper_text)) - _REQ_ID_EXCLUDE

    # 언더스코어 형태 → 하이픈으로 정규화
    for prefix, num in _CODE_REQ_PATTERN.findall(text):
        normalized = f"{prefix.upper()}-{num}"
        if normalized not in _REQ_ID_EXCLUDE:
            req_ids.add(normalized)

    return sorted(req_ids)


def _generate_function_id(file_path: str, signature: str) -> str:
    """파일 경로 + 시그니처로 고유한 function_id를 생성한다.

    동일 파일의 동일 함수에 대해 항상 같은 ID가 반환된다.
    서브 청크 재조립(reassembly) 시 형제 청크를 식별하는 키로 사용.
    """
    key = f"{file_path}::{signature}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _build_file_summary_chunk(
    rel_path: str, source_bytes: bytes, root_node, lang_config: dict,
) -> Document:
    """파일 전체의 요약 청크(L1)를 생성한다.

    L1 파일 요약 청크는 파일의 전체 구조를 한눈에 파악할 수 있는 메타 정보를 담는다.
    "A.cpp 파일은 어떤 기능?" 같은 파일 단위 질의에 직접 매칭된다.

    포함 정보: 파일 경로, 언어, 라인 수, import 목록, Req ID, 선언 목록(시그니처)
    """
    all_text = source_bytes.decode('utf-8', errors='replace')
    lang_name = lang_config["name"]
    line_count = all_text.count('\n') + 1

    # import 문 추출
    imports = _extract_imports(root_node, source_bytes, lang_config)

    # 파일 전체에서 Req ID 추출
    req_ids = _extract_req_ids_from_code(all_text)

    # 최상위 선언 수집 (함수, 클래스, 메서드)
    target_nodes = _collect_target_nodes(root_node, lang_config, source_bytes)
    decl_lines: List[str] = []
    for node, depth, node_type in target_nodes:
        indent = "  " * (depth + 1)
        sig = _get_signature(node, source_bytes, lang_config)
        # 시그니처가 너무 길면 잘라서 표시
        if len(sig) > 120:
            sig = sig[:117] + "..."
        label = "class" if node_type == "class" else "func"
        decl_lines.append(f"{indent}- [{label}] {sig}")

    # 요약 텍스트 조합
    summary_parts = [
        f"[{rel_path} | FILE_SUMMARY]",
        f"Language: {lang_name}",
        f"Lines: {line_count}",
    ]
    if imports:
        imports_str = ', '.join(imports[:15])
        if len(imports) > 15:
            imports_str += f" ... (+{len(imports) - 15})"
        summary_parts.append(f"Imports: {imports_str}")
    if req_ids:
        ids_str = ', '.join(req_ids[:30])
        if len(req_ids) > 30:
            ids_str += f" ... (+{len(req_ids) - 30})"
        summary_parts.append(f"Requirement IDs: {ids_str}")
    if decl_lines:
        summary_parts.append("Declarations:")
        summary_parts.extend(decl_lines[:100])  # 최대 100개 선언
        if len(decl_lines) > 100:
            summary_parts.append(f"  ... (+{len(decl_lines) - 100} more)")

    return Document(
        page_content='\n'.join(summary_parts),
        metadata={
            "source": rel_path,
            "language": lang_name,
            "chunk_type": "file_summary",
            "requirement_ids": req_ids,
            "line_range": (1, line_count),
        },
    )


def _chunk_code_file_ts(
    file_path: str, rel_path: str, source_bytes: bytes,
    lang_config: dict, root_node,
    max_chunk_size: int = 4000,
    min_aggregate_size: int = 200,
) -> List[Document]:
    """tree-sitter AST를 사용하여 소스 파일을 함수/클래스 단위로 청킹한다.

    청크 계층:
    - L2: 함수/클래스 단위 청크 (AST 노드의 정확한 바이트 범위 사용)
    - L3: 대형 함수의 서브 청크 (max_chunk_size 초과 시 AST 블록 경계로 분할)

    소형 선언(min_aggregate_size 미만)은 인접한 것끼리 병합하여
    벡터 DB가 사소한 청크로 오염되는 것을 방지한다.
    """
    all_text = source_bytes.decode('utf-8', errors='replace')
    target_nodes = _collect_target_nodes(root_node, lang_config, source_bytes)

    chunks: List[Document] = []
    small_buffer: List[str] = []  # 소형 선언 병합 버퍼
    small_meta_buf: dict = {}     # 병합 중인 메타데이터

    def flush_small_buffer() -> None:
        """소형 선언 버퍼를 하나의 청크로 병합하여 방출"""
        nonlocal small_buffer, small_meta_buf
        if not small_buffer:
            return
        merged_text = '\n\n'.join(small_buffer)
        prefix = f"[{rel_path} | declarations]"
        chunks.append(Document(
            page_content=f"{prefix}\n{merged_text}",
            metadata={
                "source": rel_path,
                "language": lang_config["name"],
                "chunk_type": "declarations",
                "hierarchy": small_meta_buf.get("hierarchy", ""),
                "signature": "(aggregated small declarations)",
                "requirement_ids": small_meta_buf.get("requirement_ids", []),
                "line_range": small_meta_buf.get("line_range", (0, 0)),
                "symbols_defined": small_meta_buf.get("symbols_defined", []),
            },
        ))
        small_buffer = []
        small_meta_buf = {}

    for node, depth, node_type in target_nodes:
        code_text = source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
        start_line = node.start_point[0] + 1  # 0-based → 1-based
        end_line = node.end_point[0] + 1
        name = _get_node_name(node, source_bytes) or "(anonymous)"
        hierarchy = _extract_hierarchy(node, source_bytes, lang_config)
        signature = _get_signature(node, source_bytes, lang_config)
        req_ids = _extract_req_ids_from_code(code_text)

        # ── 소형 선언 병합 ──
        if len(code_text) < min_aggregate_size and node_type == "function":
            small_buffer.append(code_text)
            if not small_meta_buf:
                small_meta_buf = {
                    "hierarchy": hierarchy,
                    "requirement_ids": list(req_ids),
                    "line_range": (start_line, end_line),
                    "symbols_defined": [name],
                }
            else:
                small_meta_buf["requirement_ids"] = sorted(
                    set(small_meta_buf["requirement_ids"] + list(req_ids))
                )
                small_meta_buf["line_range"] = (
                    small_meta_buf["line_range"][0], end_line
                )
                small_meta_buf["symbols_defined"].append(name)
            # 버퍼가 충분히 쌓이면 방출
            if sum(len(s) for s in small_buffer) >= min_aggregate_size * 3:
                flush_small_buffer()
            continue

        # 소형 버퍼가 있으면 먼저 방출
        flush_small_buffer()

        # ── L2/L3 청크 생성 ──
        hierarchy_with_name = f"{hierarchy} > {name}" if hierarchy else name
        prefix = f"[{rel_path} | {hierarchy_with_name}]"
        full_content = f"{prefix}\n{code_text}"

        if len(full_content) <= max_chunk_size:
            # ── L2: 함수 크기가 max_chunk_size 이하 → 단일 청크 ──
            chunks.append(Document(
                page_content=full_content,
                metadata={
                    "source": rel_path,
                    "language": lang_config["name"],
                    "chunk_type": "function" if node_type != "class" else "class",
                    "hierarchy": hierarchy_with_name,
                    "signature": signature,
                    "requirement_ids": req_ids,
                    "line_range": (start_line, end_line),
                    "symbols_defined": [name],
                    "is_subchunk": False,
                },
            ))
        else:
            # ── L3: 대형 함수 → 서브 청크 분할 ──
            # RecursiveCharacterTextSplitter로 분할 후 function_id로 연결
            func_id = _generate_function_id(rel_path, signature)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=200,
                separators=["\n\n", "\n    \n", "\n", " ", ""],
            )
            sub_texts = splitter.split_text(code_text)

            for idx, sub_text in enumerate(sub_texts):
                sub_prefix = f"[{rel_path} | {hierarchy_with_name} | part {idx + 1}/{len(sub_texts)}]"
                # 서브 청크에도 시그니처를 주입하여 문맥 유지
                if idx > 0:
                    sub_content = f"{sub_prefix}\n// signature: {signature}\n{sub_text}"
                else:
                    sub_content = f"{sub_prefix}\n{sub_text}"

                chunks.append(Document(
                    page_content=sub_content,
                    metadata={
                        "source": rel_path,
                        "language": lang_config["name"],
                        "chunk_type": "subchunk",
                        "hierarchy": hierarchy_with_name,
                        "signature": signature,
                        "requirement_ids": _extract_req_ids_from_code(sub_text),
                        "line_range": (start_line, end_line),
                        "symbols_defined": [name],
                        "is_subchunk": True,
                        "function_id": func_id,
                        "chunk_index": f"{idx + 1}/{len(sub_texts)}",
                    },
                ))

    # 마지막 소형 버퍼 방출
    flush_small_buffer()

    return chunks


# ── regex 폴백: tree-sitter 미설치 시 사용 ─────────────────────
# 함수/클래스 선언을 감지하는 범용 정규식 패턴
# 완벽하지 않지만, tree-sitter 없이도 기본적인 코드 청킹이 가능하다.
_FUNC_PATTERN_C = re.compile(
    r'^(?:(?:static|inline|extern|virtual|override|const|unsigned|signed|volatile|'
    r'void|int|char|float|double|long|short|bool|auto|size_t|uint\w+|int\w+)\s+)*'
    r'(?:\w+(?:::\w+)*\s+)*'  # 반환 타입 + 네임스페이스
    r'(\w+(?:::\w+)*)\s*\(',   # 함수명 + 여는 괄호
    re.MULTILINE
)
_FUNC_PATTERN_PY = re.compile(r'^((?:@\w+.*\n)*)\s*((?:async\s+)?def\s+\w+)', re.MULTILINE)
_CLASS_PATTERN_PY = re.compile(r'^(class\s+\w+)', re.MULTILINE)


def _chunk_code_file_regex(
    file_path: str, rel_path: str, content: str, lang_config: dict,
    max_chunk_size: int = 4000,
) -> List[Document]:
    """regex 기반 폴백 청킹: tree-sitter 미설치 시 사용.

    빈 줄로 구분된 블록 단위로 분할하며, 함수/클래스 선언 패턴을 감지하여
    청크 메타데이터에 시그니처를 기록한다.

    tree-sitter 대비 정확도가 떨어지므로, tree-sitter 설치를 권장한다.
    """
    lang_name = lang_config["name"]

    # 빈 줄 2개 이상으로 분할 (함수 사이의 자연스러운 경계)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=200,
        separators=["\n\n\n", "\n\n", "\n", " ", ""],
    )
    texts = splitter.split_text(content)

    chunks: List[Document] = []
    for idx, text in enumerate(texts):
        # 간이 시그니처 추출
        signature = ""
        if lang_name == "python":
            m = _FUNC_PATTERN_PY.search(text)
            if m:
                signature = m.group(2).strip()
            else:
                m = _CLASS_PATTERN_PY.search(text)
                if m:
                    signature = m.group(1).strip()
        else:
            m = _FUNC_PATTERN_C.search(text)
            if m:
                signature = text[m.start():text.find('\n', m.start())].strip()

        req_ids = _extract_req_ids_from_code(text)
        prefix = f"[{rel_path} | chunk {idx + 1}]"
        if signature:
            prefix = f"[{rel_path} | {signature}]"

        start_line = content[:content.find(text[:50])].count('\n') + 1 if text[:50] in content else 0

        chunks.append(Document(
            page_content=f"{prefix}\n{text}",
            metadata={
                "source": rel_path,
                "language": lang_name,
                "chunk_type": "function",
                "hierarchy": "",
                "signature": signature,
                "requirement_ids": req_ids,
                "line_range": (start_line, start_line + text.count('\n')),
                "symbols_defined": [],
                "is_subchunk": False,
            },
        ))

    return chunks


def _build_file_summary_regex(
    rel_path: str, content: str, lang_config: dict,
) -> Document:
    """regex 기반 파일 요약 청크(L1) 생성 — tree-sitter 미설치 시 폴백."""
    lang_name = lang_config["name"]
    line_count = content.count('\n') + 1
    req_ids = _extract_req_ids_from_code(content)

    # 간이 import 추출
    imports: List[str] = []
    for line in content.split('\n')[:100]:  # 상위 100줄에서만 추출
        stripped = line.strip()
        if lang_name == "python" and (stripped.startswith('import ') or stripped.startswith('from ')):
            imports.append(stripped)
        elif lang_name in ("c", "cpp") and stripped.startswith('#include'):
            imports.append(stripped)
        elif lang_name == "java" and stripped.startswith('import '):
            imports.append(stripped)

    summary_parts = [
        f"[{rel_path} | FILE_SUMMARY]",
        f"Language: {lang_name}",
        f"Lines: {line_count}",
    ]
    if imports:
        summary_parts.append(f"Imports: {', '.join(imports[:15])}")
    if req_ids:
        summary_parts.append(f"Requirement IDs: {', '.join(req_ids[:30])}")

    return Document(
        page_content='\n'.join(summary_parts),
        metadata={
            "source": rel_path,
            "language": lang_name,
            "chunk_type": "file_summary",
            "requirement_ids": req_ids,
            "line_range": (1, line_count),
        },
    )


# ══════════════════════════════════════════════════════════════
# BaseRAG — 공통 인프라 (임베딩, FAISS/BM25, 앙상블 검색)
# ══════════════════════════════════════════════════════════════

class BaseRAG:
    """MarkdownRAG와 CodeRAG의 공통 기반 클래스.

    임베딩 모델 초기화, FAISS/BM25 벡터 스토어 관리, 앙상블 검색 설정 등
    양쪽 RAG에서 동일하게 사용하는 인프라를 제공한다.

    .env 환경 변수:
    - EMBEDDING_PROVIDER: "ollama" 또는 "gemini"
    - OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL: Ollama 설정
    - GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL: Gemini 설정
    - CODE_EMBEDDING_MODEL: 코드 전용 임베딩 모델 (미설정 시 기본 모델 사용)
    """

    def __init__(self, db_store_path: str = "./knowledge_base",
                 embedding_model_override: Optional[str] = None):
        """BaseRAG 초기화.

        Args:
            db_store_path: DB를 저장/로드할 디렉토리 경로
            embedding_model_override: 기본 임베딩 모델 대신 사용할 모델명
                                      (CodeRAG에서 CODE_EMBEDDING_MODEL 적용 시 사용)
        """
        load_dotenv()
        self.db_store_path = db_store_path
        self.faiss_path = os.path.join(self.db_store_path, "faiss_index")
        self.bm25_path = os.path.join(self.db_store_path, "bm25_retriever.pkl")

        self.vector_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None

        self.embeddings = self._init_embeddings(embedding_model_override)

    def _init_embeddings(self, model_override: Optional[str] = None):
        """임베딩 모델을 초기화한다.

        .env의 EMBEDDING_PROVIDER에 따라 Ollama 또는 Gemini 임베딩을 사용.
        model_override가 지정되면 기본 모델 대신 해당 모델을 사용한다.
        """
        provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
        if provider == "gemini":
            if GoogleGenerativeAIEmbeddings is None:
                raise ImportError("langchain-google-genai 패키지가 설치되지 않았습니다.")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")
            model = model_override or os.getenv("GEMINI_EMBEDDING_MODEL")
            if not model:
                raise ValueError(".env 파일에 GEMINI_EMBEDDING_MODEL 값이 설정되지 않았습니다.")
            return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        else:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = model_override or os.getenv("OLLAMA_EMBEDDING_MODEL")
            if not model:
                raise ValueError(".env 파일에 OLLAMA_EMBEDDING_MODEL 값이 설정되지 않았습니다.")
            return OllamaEmbeddings(base_url=base_url, model=model)

    def _setup_ensemble(self, faiss_k: int = 5, bm25_k: int = 5,
                        weights: Optional[List[float]] = None):
        """FAISS와 BM25를 묶어 Hybrid Retriever를 구성한다.

        Args:
            faiss_k: FAISS에서 반환할 결과 수
            bm25_k: BM25에서 반환할 결과 수
            weights: [BM25_가중치, FAISS_가중치] (기본: [0.4, 0.6])
        """
        if weights is None:
            weights = [0.4, 0.6]
        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": faiss_k})
        self.bm25_retriever.k = bm25_k
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=weights,
        )

    def _save_vector_stores(self, all_splits: List[Document],
                            bm25_preprocess_func=None):
        """FAISS 벡터 DB와 BM25 인덱스를 구축하고 디스크에 저장한다.

        Args:
            all_splits: 인덱싱할 Document 리스트
            bm25_preprocess_func: BM25 전처리 함수 (마크다운/코드별 다름)
        """
        print(f"총 {len(all_splits)}개의 청크(Chunk)로 분할되었습니다. 임베딩 진행 중...")
        os.makedirs(self.db_store_path, exist_ok=True)

        self.vector_store = FAISS.from_documents(all_splits, self.embeddings)
        self.vector_store.save_local(self.faiss_path)

        kwargs = {}
        if bm25_preprocess_func:
            kwargs["preprocess_func"] = bm25_preprocess_func
        self.bm25_retriever = BM25Retriever.from_documents(all_splits, **kwargs)

        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.bm25_retriever, f)

    def _load_vector_stores(self):
        """디스크에서 FAISS 벡터 DB와 BM25 인덱스를 로드한다."""
        self.vector_store = FAISS.load_local(
            self.faiss_path, self.embeddings, allow_dangerous_deserialization=True,
        )
        with open(self.bm25_path, "rb") as f:
            self.bm25_retriever = pickle.load(f)


# ══════════════════════════════════════════════════════════════
# MarkdownRAG — 마크다운 기술문서 전용 RAG
# ══════════════════════════════════════════════════════════════

class MarkdownRAG(BaseRAG):
    """FAISS + BM25 하이브리드 RAG (마크다운 기술문서 전용).

    헤더 경계 기반 구조적 청킹, Requirement ID/약어 용어 인덱스,
    희귀 용어 직접 검색 + 앙상블 보충 전략을 제공한다.
    """

    def __init__(self, db_store_path: str = "./knowledge_base",
                 doc_name: str = ""):
        """MarkdownRAG 초기화.

        Args:
            db_store_path: DB를 저장/로드할 디렉토리 경로
            doc_name: 문서 식별명 (예: "OCP 2.6 Spec", "NVMe 2.3 Base")
                      미지정 시 소스 파일/폴더명에서 자동 추출.
                      Fan-out 검색 시 결과가 어느 문서에서 왔는지 구분하는 데 사용.
        """
        super().__init__(db_store_path)
        self.term_index_path = os.path.join(self.db_store_path, "term_index.pkl")
        self.doc_meta_path = os.path.join(self.db_store_path, "doc_meta.pkl")
        self.term_index: dict = {}  # {용어(대문자): [Document, ...]} — Req ID + 약어 통합
        self.doc_name: str = doc_name

    def is_db_exists(self) -> bool:
        """FAISS DB, BM25 캐시, 용어 인덱스가 모두 존재하는지 확인"""
        return (os.path.exists(self.faiss_path)
                and os.path.exists(self.bm25_path)
                and os.path.exists(self.term_index_path))

    def load_db(self):
        """기존 구축된 DB 로드 (FAISS + BM25 + 용어 인덱스 + 문서 메타)"""
        print(f"📦 로컬에 구축된 DB를 '{self.db_store_path}'에서 로드 중...")
        self._load_vector_stores()
        with open(self.term_index_path, "rb") as f:
            self.term_index = pickle.load(f)
        # 문서 메타 정보 복원
        if os.path.exists(self.doc_meta_path):
            with open(self.doc_meta_path, "rb") as f:
                meta = pickle.load(f)
                if not self.doc_name:
                    self.doc_name = meta.get("doc_name", "")
        self._setup_ensemble()
        display_name = self.doc_name or "(미지정)"
        print(f"✅ DB 재로드 완료! (문서: {display_name}, "
              f"용어 인덱스: {len(self.term_index)}종)")

    def build_db_from_files(self, file_paths: List[str]):
        """마크다운 파일 목록을 헤더 경계 기준 청킹 후 Vector DB와 BM25 구축"""
        print(f"🔨 {len(file_paths)}개의 파일로 DB 구축 시작...")

        all_splits: List[Document] = []
        for path in file_paths:
            if not os.path.exists(path):
                print(f"⚠️ 경고: '{path}' 파일을 찾을 수 없습니다.")
                continue
            source = os.path.basename(path)
            print(f"  📄 마크다운 로딩 중: {source}")
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            splits = _split_md_by_header_boundary(content, source)
            all_splits.extend(splits)

        if not all_splits:
            raise ValueError("문서에서 추출된 텍스트가 없습니다. 유효한 마크다운 파일인지 확인하세요.")

        # 모든 청크에 문서 식별 메타데이터 주입
        # Fan-out 검색 시 결과가 어느 문서에서 왔는지 구분하는 데 사용
        for doc in all_splits:
            doc.metadata["doc_name"] = self.doc_name

        self._save_vector_stores(all_splits, bm25_preprocessor)

        # 문서 메타 정보 저장 (DB 재로드 시 문서명 복원에 사용)
        os.makedirs(self.db_store_path, exist_ok=True)
        with open(self.doc_meta_path, "wb") as f:
            pickle.dump({"doc_name": self.doc_name}, f)

        # 용어 인덱스 구축: Requirement ID + 약어 정의를 통합
        self.term_index = {}
        for doc in all_splits:
            for req_id in doc.metadata.get("requirement_ids", []):
                self.term_index.setdefault(req_id, []).append(doc)
            for abbr, full_name in doc.metadata.get("abbreviations", {}).items():
                self.term_index.setdefault(abbr, []).append(doc)
                self.term_index.setdefault(full_name.upper(), []).append(doc)
        with open(self.term_index_path, "wb") as f:
            pickle.dump(self.term_index, f)

        self._setup_ensemble()
        print(f"✅ DB 구축 완료! (청크: {len(all_splits)}, 용어 인덱스: {len(self.term_index)}종)")
        print(f"   저장 경로: '{self.db_store_path}'")

    def build_or_load(self, source_path: str):
        """DB가 로컬에 있으면 즉시 로드, 없으면 source_path를 인덱싱하여 새로 구축"""
        if self.is_db_exists():
            self.load_db()
        else:
            self._build_from_source(source_path)

    def _build_from_source(self, source_path: str):
        """단일 .md 파일 또는 폴더 내 .md 파일들을 인덱싱하여 DB 구축"""
        source_path = os.path.abspath(os.path.expanduser(source_path))

        # 문서명 자동 추출 (미지정 시 파일명/폴더명 사용)
        if not self.doc_name:
            self.doc_name = os.path.splitext(os.path.basename(source_path))[0]

        if os.path.isfile(source_path):
            self.build_db_from_files([source_path])
        elif os.path.isdir(source_path):
            exclude_dirs = {'.venv', 'venv', 'node_modules', '__pycache__', '.git',
                            '.claude', 'build', 'dist', '.vscode', '.idea'}
            files = []
            for root, dirs, filenames in os.walk(source_path):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                for f in filenames:
                    if f.lower().endswith('.md'):
                        files.append(os.path.join(root, f))
            if not files:
                raise FileNotFoundError(f"'{source_path}' 내에 인덱싱할 .md 파일이 없습니다.")
            self.build_db_from_files(files)
        else:
            raise FileNotFoundError(f"'{source_path}' 파일 또는 폴더를 찾을 수 없습니다.")

    def _extract_terms(self, query: str) -> set:
        """쿼리에서 용어 인덱스와 매칭 가능한 키를 추출.

        1. Requirement ID 패턴: TEL-6, SEC-3 등
        2. 대문자 약어: RESERVS, MO, SQ 등 (2글자 이상)
        3. 대문자 시작 복합 단어: "Submission Queue", "Management Operation" 등
        모두 대문자로 정규화하여 term_index 키와 매칭.
        """
        terms: set = set()
        upper_query = query.upper()
        terms.update(set(_REQ_ID_RE.findall(upper_query)) - _REQ_ID_EXCLUDE)
        for _, abbr in _ABBR_DEF_RE.findall(query):
            terms.add(abbr)
        for word in re.findall(r'\b([A-Z][A-Z0-9]{1,})\b', query):
            terms.add(word)
        for full_name in re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+)+)', query):
            key = full_name.upper()
            if key in self.term_index:
                terms.add(key)
        return terms

    # 용어 인덱스 직접 검색의 최대 청크 수 임계값.
    TERM_INDEX_MAX_HITS = 10

    def retrieve(self, query: str, top_k: int = 4) -> List[Document]:
        """주어진 쿼리에 대해 하이브리드 검색(의미+키워드) 결과 반환.

        쿼리에 구조화된 용어(Requirement ID, 약어, 기술 용어)가 포함되면:
        1. 용어 인덱스에서 희귀 용어만 직접 검색 (매핑 청크 수 ≤ TERM_INDEX_MAX_HITS)
        2. 앙상블 검색으로 시맨틱/키워드 보충
        3. 직접 매칭 우선 배치 + 앙상블 보충 + 중복 제거
        """
        if not self.ensemble_retriever:
            raise RuntimeError("검색할 DB가 로드되지 않았습니다. build_or_load()를 먼저 호출하세요.")

        query_terms = self._extract_terms(query)
        rare_terms = {t for t in query_terms
                      if len(self.term_index.get(t, [])) <= self.TERM_INDEX_MAX_HITS}

        if rare_terms:
            term_matched: List[Document] = []
            seen_contents: set = set()
            for term in rare_terms:
                for doc in self.term_index.get(term, []):
                    key = doc.page_content[:150]
                    if key not in seen_contents:
                        term_matched.append(doc)
                        seen_contents.add(key)

            self.bm25_retriever.k = top_k
            self.vector_store.override_search_kwargs = {"k": top_k}
            ensemble_results = self.ensemble_retriever.invoke(query)

            for doc in ensemble_results:
                key = doc.page_content[:150]
                if key not in seen_contents:
                    term_matched.append(doc)
                    seen_contents.add(key)

            return term_matched[:top_k]
        else:
            self.bm25_retriever.k = top_k
            self.vector_store.override_search_kwargs = {"k": top_k}
            return self.ensemble_retriever.invoke(query)[:top_k]


# ══════════════════════════════════════════════════════════════
# CodeRAG — 소스코드 전용 RAG
# ══════════════════════════════════════════════════════════════

class CodeRAG(BaseRAG):
    """FAISS + BM25 하이브리드 RAG (소스코드 전용).

    tree-sitter AST 기반 구조적 청킹으로 함수/클래스 경계를 정확히 보존한다.
    3단계 청크 체계(L1 파일 요약, L2 함수/클래스, L3 서브 청크)와
    5종 인덱스(symbol, req_id, file_path, function_id, reference)를 통해
    심볼 정확 매칭 + 시맨틱 검색 + 서브 청크 재조립을 수행한다.

    지원 언어: Python, C, C++, Java (tree-sitter 그래머 설치 시)
    tree-sitter 미설치 시 regex 기반 폴백으로 동작한다.

    검색 패턴:
    - "TEL-2 관련 평가를 진행하고 싶어" → req_id_index로 코드 내 TEL-2 정확 매칭
    - "A.cpp 파일은 어떤 기능?" → file_path_index로 L1 파일 요약 직접 반환
    - "allocate_block" → symbol_index로 함수 정의 직접 반환
    - "메모리 할당 로직" → Ensemble 시맨틱+키워드 검색
    """

    # 지원하는 소스코드 확장자 (이 확장자의 파일만 인덱싱)
    CODE_EXTENSIONS: Set[str] = set(_LANGUAGE_CONFIG.keys())

    def __init__(self, db_store_path: str = "./knowledge_base",
                 project_name: str = ""):
        """CodeRAG 초기화.

        Args:
            db_store_path: DB를 저장/로드할 디렉토리 경로
            project_name: 프로젝트 식별명 (예: "DeviceA", "NVMe_Test_FW")
                          미지정 시 source_path의 폴더명에서 자동 추출.
                          장비별 코드 프로젝트 구분 및 Fan-out 검색 결과 출처 표시에 사용.
        """
        # CODE_EMBEDDING_MODEL이 설정되어 있으면 코드 전용 임베딩 사용
        code_model = os.getenv("CODE_EMBEDDING_MODEL")
        super().__init__(db_store_path, embedding_model_override=code_model)

        # 프로젝트 식별 정보 (DB에 저장되어 검색 결과에서 출처 구분에 사용)
        self.project_name: str = project_name
        self.project_root: str = ""  # 빌드 시 설정, DB 로드 시 복원
        self.project_meta_path = os.path.join(self.db_store_path, "project_meta.pkl")

        # ── 5종 인덱스 경로 ──
        self.symbol_index_path = os.path.join(self.db_store_path, "symbol_index.pkl")
        self.req_id_index_path = os.path.join(self.db_store_path, "req_id_index.pkl")
        self.file_path_index_path = os.path.join(self.db_store_path, "file_path_index.pkl")
        self.function_id_index_path = os.path.join(self.db_store_path, "function_id_index.pkl")
        self.file_manifest_path = os.path.join(self.db_store_path, "file_manifest.pkl")

        # ── 인덱스 인스턴스 ──
        # symbol_index: 함수/클래스명 → 해당 심볼이 정의된 청크 리스트
        self.symbol_index: Dict[str, List[Document]] = {}
        # req_id_index: Requirement ID → 해당 ID가 포함된 청크 리스트
        self.req_id_index: Dict[str, List[Document]] = {}
        # file_path_index: 파일 상대경로 → L1 파일 요약 청크
        self.file_path_index: Dict[str, Document] = {}
        # function_id_index: function_id → 서브 청크 리스트 (재조립용)
        self.function_id_index: Dict[str, List[Document]] = {}
        # file_manifest: 파일 경로 → mtime (증분 빌드용)
        self.file_manifest: Dict[str, float] = {}

    def is_db_exists(self) -> bool:
        """FAISS DB, BM25 캐시, 5종 인덱스가 모두 존재하는지 확인"""
        return (os.path.exists(self.faiss_path)
                and os.path.exists(self.bm25_path)
                and os.path.exists(self.symbol_index_path)
                and os.path.exists(self.file_path_index_path))

    def load_db(self):
        """기존 구축된 DB 로드 (FAISS + BM25 + 5종 인덱스 + 프로젝트 메타)"""
        print(f"📦 로컬에 구축된 CodeRAG DB를 '{self.db_store_path}'에서 로드 중...")
        self._load_vector_stores()

        # 프로젝트 메타 정보 복원 (프로젝트명, 루트 경로)
        if os.path.exists(self.project_meta_path):
            with open(self.project_meta_path, "rb") as f:
                meta = pickle.load(f)
                # 명시적으로 지정하지 않은 경우에만 DB에서 복원
                if not self.project_name:
                    self.project_name = meta.get("project_name", "")
                self.project_root = meta.get("project_root", "")

        # 인덱스 로드 (각각 존재 여부 확인 후 로드)
        for attr, path in [
            ("symbol_index", self.symbol_index_path),
            ("req_id_index", self.req_id_index_path),
            ("file_path_index", self.file_path_index_path),
            ("function_id_index", self.function_id_index_path),
            ("file_manifest", self.file_manifest_path),
        ]:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    setattr(self, attr, pickle.load(f))

        self._setup_ensemble()
        print(f"✅ CodeRAG DB 재로드 완료! "
              f"(프로젝트: {self.project_name or '(미지정)'}, "
              f"심볼: {len(self.symbol_index)}, Req ID: {len(self.req_id_index)}, "
              f"파일: {len(self.file_path_index)})")

    def build_or_load(self, source_path: str):
        """DB가 로컬에 있으면 즉시 로드, 없으면 source_path를 인덱싱하여 새로 구축"""
        if self.is_db_exists():
            self.load_db()
        else:
            self._build_from_source(source_path)

    def _build_from_source(self, source_path: str):
        """소스코드 파일/폴더를 인덱싱하여 DB 구축.

        지원 확장자(.py, .c, .cpp, .h, .hpp, .cc, .java)에 해당하는 파일만 수집.
        빌드 제외 폴더(.venv, node_modules, .git, build 등)는 자동 스킵.
        """
        source_path = os.path.abspath(os.path.expanduser(source_path))

        if os.path.isfile(source_path):
            files = [source_path]
        elif os.path.isdir(source_path):
            exclude_dirs = {'.venv', 'venv', 'node_modules', '__pycache__', '.git',
                            '.claude', 'build', 'dist', '.vscode', '.idea',
                            'out', 'target', '.gradle', 'cmake-build-debug'}
            files = []
            for root, dirs, filenames in os.walk(source_path):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                for fname in filenames:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in self.CODE_EXTENSIONS:
                        files.append(os.path.join(root, fname))
        else:
            raise FileNotFoundError(f"'{source_path}' 파일 또는 폴더를 찾을 수 없습니다.")

        if not files:
            raise FileNotFoundError(f"'{source_path}' 내에 인덱싱할 소스코드 파일이 없습니다.\n"
                                    f"지원 확장자: {', '.join(sorted(self.CODE_EXTENSIONS))}")

        # 프로젝트 루트 결정 (상대 경로 생성용)
        project_root = source_path if os.path.isdir(source_path) else os.path.dirname(source_path)
        self.project_root = project_root

        # 프로젝트명 자동 추출 (미지정 시 폴더명 사용)
        if not self.project_name:
            self.project_name = os.path.basename(project_root)

        self._build_db_from_files(files, project_root)

    def _build_db_from_files(self, file_paths: List[str], project_root: str):
        """소스코드 파일 목록을 AST 기반 청킹 후 Vector DB와 인덱스 구축"""
        print(f"🔨 {len(file_paths)}개의 소스 파일로 CodeRAG DB 구축 시작...")

        # tree-sitter 사용 가능 여부 안내
        if _TREE_SITTER_AVAILABLE:
            supported = [ext for ext in sorted(self.CODE_EXTENSIONS) if ext in _TS_LANGUAGES]
            fallback = [ext for ext in sorted(self.CODE_EXTENSIONS) if ext not in _TS_LANGUAGES]
            print(f"  🌳 tree-sitter 파싱: {', '.join(supported) if supported else '없음'}")
            if fallback:
                print(f"  📝 regex 폴백: {', '.join(fallback)}")
        else:
            print("  ⚠️ tree-sitter 미설치 — 모든 파일에 regex 폴백 적용")
            print("  💡 정확한 AST 청킹을 위해 tree-sitter 설치를 권장합니다:")
            print("     pip install tree-sitter tree-sitter-python tree-sitter-c tree-sitter-cpp tree-sitter-java")

        all_splits: List[Document] = []

        for path in file_paths:
            if not os.path.exists(path):
                print(f"  ⚠️ 경고: '{path}' 파일을 찾을 수 없습니다.")
                continue

            ext = os.path.splitext(path)[1].lower()
            lang_config = _LANGUAGE_CONFIG.get(ext)
            if not lang_config:
                continue

            rel_path = os.path.relpath(path, project_root)
            print(f"  📄 {rel_path} ({lang_config['name']})")

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                print(f"  ⚠️ 읽기 실패: {e}")
                continue

            source_bytes = content.encode('utf-8')

            # tree-sitter 또는 regex로 청킹
            parser = _get_ts_parser(ext)
            if parser is not None:
                tree = parser.parse(source_bytes)
                # L1: 파일 요약 청크
                file_summary = _build_file_summary_chunk(
                    rel_path, source_bytes, tree.root_node, lang_config)
                all_splits.append(file_summary)
                # L2/L3: 함수/클래스 청크 + 서브 청크
                code_chunks = _chunk_code_file_ts(
                    path, rel_path, source_bytes, lang_config, tree.root_node)
                all_splits.extend(code_chunks)
            else:
                # regex 폴백
                file_summary = _build_file_summary_regex(rel_path, content, lang_config)
                all_splits.append(file_summary)
                code_chunks = _chunk_code_file_regex(path, rel_path, content, lang_config)
                all_splits.extend(code_chunks)

            # 파일 매니페스트 기록 (증분 빌드용)
            self.file_manifest[path] = os.path.getmtime(path)

        if not all_splits:
            raise ValueError("소스코드에서 추출된 청크가 없습니다.")

        # 모든 청크에 프로젝트 식별 메타데이터 주입
        # Fan-out 검색 시 결과가 어느 프로젝트에서 왔는지 구분하는 데 사용
        for doc in all_splits:
            doc.metadata["project"] = self.project_name

        # FAISS + BM25 구축
        self._save_vector_stores(all_splits, code_bm25_preprocessor)

        # ── 5종 인덱스 구축 ──
        self.symbol_index = {}
        self.req_id_index = {}
        self.file_path_index = {}
        self.function_id_index = {}

        for doc in all_splits:
            chunk_type = doc.metadata.get("chunk_type", "")

            # 파일 요약 인덱스
            if chunk_type == "file_summary":
                source = doc.metadata.get("source", "")
                self.file_path_index[source] = doc
                # 파일명만으로도 검색 가능하도록 추가 등록
                basename = os.path.basename(source)
                self.file_path_index[basename] = doc

            # 심볼 인덱스 (함수명, 클래스명)
            for sym in doc.metadata.get("symbols_defined", []):
                self.symbol_index.setdefault(sym, []).append(doc)
                # snake_case/CamelCase 분리 키도 등록
                # "allocate_block" → "allocate_block" (원형) + 개별 단어는 BM25에 맡김

            # Requirement ID 인덱스
            for req_id in doc.metadata.get("requirement_ids", []):
                self.req_id_index.setdefault(req_id, []).append(doc)

            # function_id 인덱스 (서브 청크 재조립용)
            func_id = doc.metadata.get("function_id")
            if func_id:
                self.function_id_index.setdefault(func_id, []).append(doc)

        # 인덱스 + 프로젝트 메타 저장
        os.makedirs(self.db_store_path, exist_ok=True)

        # 프로젝트 메타 정보 저장 (DB 재로드 시 프로젝트명/루트 경로 복원에 사용)
        with open(self.project_meta_path, "wb") as f:
            pickle.dump({
                "project_name": self.project_name,
                "project_root": self.project_root,
            }, f)

        for attr, path in [
            ("symbol_index", self.symbol_index_path),
            ("req_id_index", self.req_id_index_path),
            ("file_path_index", self.file_path_index_path),
            ("function_id_index", self.function_id_index_path),
            ("file_manifest", self.file_manifest_path),
        ]:
            with open(path, "wb") as f:
                pickle.dump(getattr(self, attr), f)

        self._setup_ensemble()
        print(f"✅ CodeRAG DB 구축 완료!")
        print(f"   청크: {len(all_splits)} | 심볼: {len(self.symbol_index)} | "
              f"Req ID: {len(self.req_id_index)} | 파일: {len(self.file_path_index)}")
        print(f"   저장 경로: '{self.db_store_path}'")

    # 용어/심볼 인덱스 직접 검색의 최대 청크 수 임계값.
    TERM_INDEX_MAX_HITS = 10

    def _extract_code_terms(self, query: str) -> dict:
        """쿼리에서 검색 가능한 용어를 추출한다.

        추출 대상:
        - file_paths: "A.cpp", "main.py" 등 파일명 패턴
        - req_ids: TEL-2, SEC-5 등 Requirement ID
        - symbols: 함수명, 클래스명 등 코드 식별자

        반환: {"file_paths": set, "req_ids": set, "symbols": set}
        """
        terms = {"file_paths": set(), "req_ids": set(), "symbols": set()}
        upper_query = query.upper()

        # 파일 경로 패턴 감지 (확장자 포함 단어)
        for word in re.findall(r'[\w./\\-]+\.\w{1,5}', query):
            ext = os.path.splitext(word)[1].lower()
            if ext in self.CODE_EXTENSIONS or ext == '.md':
                terms["file_paths"].add(word)

        # Requirement ID (하이픈 형태)
        terms["req_ids"].update(set(_REQ_ID_RE.findall(upper_query)) - _REQ_ID_EXCLUDE)

        # Requirement ID (언더스코어 형태)
        for prefix, num in _CODE_REQ_PATTERN.findall(query):
            normalized = f"{prefix.upper()}-{num}"
            if normalized not in _REQ_ID_EXCLUDE:
                terms["req_ids"].add(normalized)

        # 코드 식별자 (CamelCase 또는 snake_case 패턴)
        for word in re.findall(r'\b([A-Za-z_]\w{2,})\b', query):
            # 일반 영어 단어와 구분하기 위해 코드 식별자 패턴만 선택
            if ('_' in word or  # snake_case
                re.search(r'[a-z][A-Z]', word) or  # camelCase
                word[0].isupper() and len(word) > 2):  # PascalCase
                if word in self.symbol_index:
                    terms["symbols"].add(word)

        return terms

    def _reassemble_subchunks(self, docs: List[Document]) -> List[Document]:
        """검색 결과에서 서브 청크를 감지하여 완전한 함수로 재조립한다.

        서브 청크(is_subchunk=True)가 발견되면:
        1. function_id_index에서 같은 function_id의 모든 형제 서브 청크를 수집
        2. chunk_index 순서로 정렬
        3. 하나의 Document로 병합하여 완전한 함수를 반환

        이미 재조립된 function_id는 중복 처리하지 않는다.
        """
        result: List[Document] = []
        reassembled_fids: Set[str] = set()

        for doc in docs:
            if not doc.metadata.get("is_subchunk"):
                result.append(doc)
                continue

            fid = doc.metadata.get("function_id", "")
            if fid in reassembled_fids:
                continue  # 이미 이 함수는 재조립됨

            # 같은 function_id의 모든 서브 청크 수집
            siblings = self.function_id_index.get(fid, [doc])
            # chunk_index("1/3", "2/3", ...) 기준 정렬
            siblings_sorted = sorted(
                siblings,
                key=lambda d: int(d.metadata.get("chunk_index", "1/1").split("/")[0])
            )

            # 병합: 각 서브 청크의 본문을 합침
            merged_content = '\n'.join(s.page_content for s in siblings_sorted)
            merged_req_ids = sorted(set(
                rid for s in siblings_sorted
                for rid in s.metadata.get("requirement_ids", [])
            ))

            # 재조립된 Document 생성 (첫 번째 서브 청크의 메타데이터 기반)
            merged_meta = dict(siblings_sorted[0].metadata)
            merged_meta["is_subchunk"] = False
            merged_meta["chunk_index"] = None
            merged_meta["reassembled"] = True
            merged_meta["requirement_ids"] = merged_req_ids

            result.append(Document(page_content=merged_content, metadata=merged_meta))
            reassembled_fids.add(fid)

        return result

    def retrieve(self, query: str, top_k: int = 4) -> List[Document]:
        """코드 RAG 하이브리드 검색: 심볼/파일/Req ID 직접 매칭 + 앙상블 + 서브 청크 재조립.

        검색 흐름:
        1. Tier 1: 직접 검색 — 파일 경로, Req ID, 심볼 인덱스에서 정확 매칭
        2. Tier 2: Ensemble — FAISS(60%) + BM25(40%) 시맨틱+키워드 검색
        3. Tier 3: 서브 청크 재조립 — 서브 청크가 포함되면 형제 전체를 합쳐 완전한 함수 반환
        4. 중복 제거 후 top_k개 반환
        """
        if not self.ensemble_retriever:
            raise RuntimeError("검색할 DB가 로드되지 않았습니다. build_or_load()를 먼저 호출하세요.")

        terms = self._extract_code_terms(query)
        results: List[Document] = []
        seen: Set[str] = set()

        def _add_unique(doc: Document) -> None:
            """중복 제거하며 결과에 추가"""
            key = doc.page_content[:200]
            if key not in seen:
                results.append(doc)
                seen.add(key)

        # ── Tier 1: 직접 검색 ──

        # 파일 경로 매칭 ("A.cpp 파일은 어떤 기능?")
        for fp in terms["file_paths"]:
            doc = self.file_path_index.get(fp)
            if doc:
                _add_unique(doc)

        # Req ID 매칭 ("TEL-2 관련 평가")
        for req_id in terms["req_ids"]:
            hits = self.req_id_index.get(req_id, [])
            if len(hits) <= self.TERM_INDEX_MAX_HITS:
                for doc in hits:
                    _add_unique(doc)

        # 심볼 매칭 ("allocate_block 함수")
        for sym in terms["symbols"]:
            hits = self.symbol_index.get(sym, [])
            if len(hits) <= self.TERM_INDEX_MAX_HITS:
                for doc in hits:
                    _add_unique(doc)

        # ── Tier 2: Ensemble 검색 ──
        self.bm25_retriever.k = top_k
        self.vector_store.override_search_kwargs = {"k": top_k}
        ensemble_results = self.ensemble_retriever.invoke(query)
        for doc in ensemble_results:
            _add_unique(doc)

        # ── Tier 3: 서브 청크 재조립 ──
        # 검색 결과에 서브 청크가 포함되어 있으면 완전한 함수로 재조립
        reassembled = self._reassemble_subchunks(results)

        return reassembled[:top_k]


# ══════════════════════════════════════════════════════════════
# 테스트 유틸리티
# ══════════════════════════════════════════════════════════════

def _print_results(tag: str, query: str, docs: List[Document]) -> None:
    """검색 결과를 포맷팅하여 출력"""
    print(f"\n{'='*60}")
    print(f"[{tag}] 🔍 쿼리: {query}")
    print('='*60)
    for i, doc in enumerate(docs):
        meta = doc.metadata
        chunk_type = meta.get("chunk_type", "")

        # 마크다운 전용 메타데이터
        section = meta.get("section", "")
        page = meta.get("page", "")

        # 코드 전용 메타데이터
        hierarchy = meta.get("hierarchy", "")
        signature = meta.get("signature", "")
        language = meta.get("language", "")
        line_range = meta.get("line_range", "")

        # 공통 메타데이터
        source = meta.get("source", "Unknown")
        req_ids = meta.get("requirement_ids", [])
        req_info = f" | IDs: {', '.join(req_ids[:20])}" if req_ids else ""
        if len(req_ids) > 20:
            req_info += f" (+{len(req_ids)-20})"

        # 출력 형식 결정
        if chunk_type in ("file_summary", "function", "class", "subchunk", "declarations"):
            # 코드 RAG 결과
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
            # 마크다운 RAG 결과
            doc_name = meta.get("doc_name", "")
            doc_info = f" | 문서: {doc_name}" if doc_name else ""
            page_info = f" | 페이지: {page}" if page else ""
            section_info = f" | 섹션: {section}" if section else ""
            print(f"  결과 {i+1}{doc_info}{page_info}{section_info}{req_info}")

        # 본문 출력 (너무 길면 요약)
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

if __name__ == "__main__":
    import sys

    # ── 테스트 설정 ──────────────────────────────────────────────
    # 각 항목의 "type" 필드로 MarkdownRAG("markdown") 또는 CodeRAG("code")를 선택.
    # python3 rag.py [markdown|code|all] 형태로 실행하여 특정 유형만 테스트할 수 있다.
    #
    # 예시:
    #   python3 rag.py             → 전체 테스트 (markdown + code)
    #   python3 rag.py markdown    → 마크다운 RAG만 테스트
    #   python3 rag.py code        → 코드 RAG만 테스트

    TESTS = [
        # ── 마크다운 RAG 테스트 ──────────────────────────────────
        {
            "type": "markdown",
            "label": "NVMe 2.3 Base Spec",
            "source": "/Users/howard/Project/Parser/Docs/NVM-Express-Base-Specification-Revision-2.3-2025.08.01-Ratified-ocr.md",
            "db_path": "/Users/howard/Project/knowledge/nvme23",
            "queries": [
                {"tag": "의미 검색",       "query": "큐(Queue)의 제출 및 완료 메커니즘은 어떻게 동작하나요?"},
                {"tag": "약어: RESERVS",  "query": "RESERVS"},
                {"tag": "약어: MO",       "query": "What is Management Operation?"},
                {"tag": "약어: SQ",       "query": "Submission Queue의 동작 원리"},
            ],
        },
        {
            "type": "markdown",
            "label": "OCP 2.6 Datacenter NVMe SSD",
            "source": "/Users/howard/Project/Parser/Docs/Datacenter NVMe SSD Specification v2.6-ocr.md",
            "db_path": "/Users/howard/Project/knowledge/ocp26",
            "queries": [
                {"tag": "내구성/DWPD",         "query": "Datacenter NVMe SSD endurance and DWPD requirements"},
                {"tag": "전력 관리",            "query": "power consumption and thermal management"},
                {"tag": "텔레메트리",            "query": "SMART health information log telemetry"},
                {"tag": "레이턴시 모니터",        "query": "latency monitor feature and bucket configuration"},
                {"tag": "에러 복구",             "query": "error recovery and unsanitize operation"},
                {"tag": "펌웨어 업데이트",        "query": "firmware update activation and commit"},
                {"tag": "네임스페이스",           "query": "namespace management and capacity allocation"},
                {"tag": "ID검색: TEL-3",         "query": "TEL-3"},
                {"tag": "ID검색: TEL-6",         "query": "TEL-6"},
                {"tag": "ID검색: SEC-5",         "query": "SEC-5"},
                {"tag": "ID검색: FWUP-5",        "query": "FWUP-5"},
                {"tag": "자연어+ID: TEL-3",      "query": "Explain to me about TEL-3 and related contents"},
                {"tag": "자연어+ID: SEC-5",      "query": "Explain to me about SEC-5 and related contents"},
            ],
        },
        # ── 코드 RAG 테스트 ──────────────────────────────────────
        # 아래 source 경로를 실제 프로젝트 경로로 변경하여 사용하세요.
        # {
        #     "type": "code",
        #     "label": "NVMe Test Framework",
        #     "source": "/path/to/your/code/project",
        #     "db_path": "/Users/howard/Project/knowledge/nvme_test_code",
        #     "queries": [
        #         {"tag": "Req ID 검색",          "query": "TEL-2 관련 평가를 진행하고 싶어"},
        #         {"tag": "도메인 시맨틱",         "query": "Telemetry 관련 평가 구현된 이력이 있으면 설명해줘"},
        #         {"tag": "내구성 변경",           "query": "내구성 강화 평가를 진행하고 싶은데 어떻게 변경하면 좋을까?"},
        #         {"tag": "파일 요약",             "query": "main.cpp 파일은 어떤 기능을 하고 있어?"},
        #         {"tag": "심볼 검색",             "query": "allocate_block"},
        #     ],
        # },
    ]

    # CLI 인자 파싱
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    valid_modes = {"all", "markdown", "code"}
    if mode not in valid_modes:
        print(f"사용법: python3 rag.py [{' | '.join(sorted(valid_modes))}]")
        print(f"  all      : 전체 테스트 (기본값)")
        print(f"  markdown : 마크다운 RAG만 테스트")
        print(f"  code     : 코드 RAG만 테스트")
        sys.exit(1)

    # 테스트 유형 필터링
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

        # ── DB 재로드 검증 (마크다운 테스트가 포함된 경우만) ──────
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

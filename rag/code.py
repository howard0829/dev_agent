"""
rag/code.py — CodeRAG (소스코드 전용 RAG)

tree-sitter AST 기반 구조적 청킹으로 함수/클래스 경계를 정확히 보존한다.
3단계 청크 체계(L1 파일 요약, L2 함수/클래스, L3 서브 청크)와
5종 인덱스를 통해 심볼 정확 매칭 + 시맨틱 검색 + 서브 청크 재조립을 수행한다.
"""

import hashlib
import logging
import os
import pickle
import re
from typing import Dict, List, Optional, Set, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CODE_EMBEDDING_MODEL
from rag.base import BaseRAG
from rag.constants import (
    _TREE_SITTER_AVAILABLE, _LANGUAGE_CONFIG,
    _TS_LANGUAGES, _TS_PARSERS,
    _REQ_ID_RE, _REQ_ID_EXCLUDE, _CODE_REQ_PATTERN,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 코드 BM25 전처리
# ══════════════════════════════════════════════════════════════

def code_bm25_preprocessor(text: str) -> str:
    """코드 BM25 전처리: CamelCase/snake_case 분리 + Req ID 하이픈 보존."""
    # 1단계: Requirement ID 하이픈 보존
    processed = _REQ_ID_RE.sub(
        lambda m: m.group(0).replace('-', '\x00') if m.group(0) not in _REQ_ID_EXCLUDE else m.group(0),
        text,
    )
    # 2단계: CamelCase 분리
    processed = re.sub(r'([a-z])([A-Z])', r'\1 \2', processed)
    processed = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', processed)
    # 3단계: 코드 노이즈 제거
    processed = re.sub(r'[^a-zA-Z0-9가-힣\s_\x00]', ' ', processed)
    # 4단계: 언더스코어를 공백으로 치환
    processed = processed.replace('_', ' ')
    # 5단계: 플레이스홀더 복원 + 소문자화
    processed = processed.replace('\x00', '-')
    return processed.lower()


# ══════════════════════════════════════════════════════════════
# tree-sitter AST 헬퍼 함수
# ══════════════════════════════════════════════════════════════

def _get_ts_parser(ext: str) -> Optional['object']:
    """확장자에 해당하는 tree-sitter 파서를 반환. 미지원 시 None."""
    if ext not in _TS_LANGUAGES:
        return None
    if ext not in _TS_PARSERS:
        from rag.constants import TSParser
        parser = TSParser(_TS_LANGUAGES[ext])
        _TS_PARSERS[ext] = parser
    return _TS_PARSERS[ext]


def _get_node_name(node, source_bytes: bytes) -> Optional[str]:
    """AST 노드에서 이름(식별자)을 추출한다."""
    name_types = {'identifier', 'name', 'type_identifier', 'field_identifier'}
    for child in node.children:
        if child.type in name_types:
            return source_bytes[child.start_byte:child.end_byte].decode('utf-8', errors='replace')
    return None


def _get_signature(node, source_bytes: bytes, lang_config: dict) -> str:
    """함수/메서드 노드에서 시그니처(선언부)를 추출한다."""
    for child in node.children:
        if child.type in lang_config.get("body_types", []):
            sig = source_bytes[node.start_byte:child.start_byte].decode('utf-8', errors='replace').strip()
            sig = ' '.join(sig.split())
            return sig
    text = source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
    return text.split('\n')[0].strip()


def _extract_hierarchy(node, source_bytes: bytes, lang_config: dict) -> str:
    """AST 노드의 부모 체인을 역추적하여 계층 경로를 추출한다."""
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
    """파일의 최상위 import/include 문을 추출한다."""
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
    """AST를 재귀적으로 탐색하여 함수/클래스 노드를 수집한다."""
    if results is None:
        results = []

    func_types = set(lang_config.get("func_types", []))
    class_types = set(lang_config.get("class_types", []))
    decorator_aware = lang_config.get("decorator_aware", False)

    if decorator_aware and node.type == "decorated_definition":
        results.append((node, depth, "decorated_function"))
        for child in node.children:
            if child.type in class_types:
                results.append((child, depth, "class"))
                for grandchild in child.children:
                    _collect_target_nodes(grandchild, lang_config, source_bytes, results, depth + 1)
        return results

    if node.type in func_types:
        if not inside_decorated:
            results.append((node, depth, "function"))

    elif node.type in class_types:
        results.append((node, depth, "class"))
        for child in node.children:
            _collect_target_nodes(child, lang_config, source_bytes, results, depth + 1,
                                 inside_decorated=False)
        return results

    is_in_decorated = inside_decorated or (decorator_aware and node.type == "decorated_definition")
    for child in node.children:
        _collect_target_nodes(child, lang_config, source_bytes, results, depth, is_in_decorated)

    return results


def _extract_req_ids_from_code(text: str) -> List[str]:
    """코드 텍스트에서 Requirement ID를 추출한다."""
    upper_text = text.upper()
    req_ids = set(_REQ_ID_RE.findall(upper_text)) - _REQ_ID_EXCLUDE
    for prefix, num in _CODE_REQ_PATTERN.findall(text):
        normalized = f"{prefix.upper()}-{num}"
        if normalized not in _REQ_ID_EXCLUDE:
            req_ids.add(normalized)
    return sorted(req_ids)


def _generate_function_id(file_path: str, signature: str) -> str:
    """파일 경로 + 시그니처로 고유한 function_id를 생성한다."""
    key = f"{file_path}::{signature}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _build_file_summary_chunk(
    rel_path: str, source_bytes: bytes, root_node, lang_config: dict,
) -> Document:
    """파일 전체의 요약 청크(L1)를 생성한다."""
    all_text = source_bytes.decode('utf-8', errors='replace')
    lang_name = lang_config["name"]
    line_count = all_text.count('\n') + 1

    imports = _extract_imports(root_node, source_bytes, lang_config)
    req_ids = _extract_req_ids_from_code(all_text)
    target_nodes = _collect_target_nodes(root_node, lang_config, source_bytes)

    decl_lines: List[str] = []
    for node, depth, node_type in target_nodes:
        indent = "  " * (depth + 1)
        sig = _get_signature(node, source_bytes, lang_config)
        if len(sig) > 120:
            sig = sig[:117] + "..."
        label = "class" if node_type == "class" else "func"
        decl_lines.append(f"{indent}- [{label}] {sig}")

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
        summary_parts.extend(decl_lines[:100])
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
    """tree-sitter AST를 사용하여 소스 파일을 함수/클래스 단위로 청킹한다."""
    all_text = source_bytes.decode('utf-8', errors='replace')
    target_nodes = _collect_target_nodes(root_node, lang_config, source_bytes)

    chunks: List[Document] = []
    small_buffer: List[str] = []
    small_meta_buf: dict = {}

    def flush_small_buffer() -> None:
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
        start_line = node.start_point[0] + 1
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
            if sum(len(s) for s in small_buffer) >= min_aggregate_size * 3:
                flush_small_buffer()
            continue

        flush_small_buffer()

        # ── L2/L3 청크 생성 ──
        hierarchy_with_name = f"{hierarchy} > {name}" if hierarchy else name
        prefix = f"[{rel_path} | {hierarchy_with_name}]"
        full_content = f"{prefix}\n{code_text}"

        if len(full_content) <= max_chunk_size:
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
            func_id = _generate_function_id(rel_path, signature)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=200,
                separators=["\n\n", "\n    \n", "\n", " ", ""],
            )
            sub_texts = splitter.split_text(code_text)

            for idx, sub_text in enumerate(sub_texts):
                sub_prefix = f"[{rel_path} | {hierarchy_with_name} | part {idx + 1}/{len(sub_texts)}]"
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

    flush_small_buffer()
    return chunks


# ── regex 폴백: tree-sitter 미설치 시 사용 ─────────────────────
_FUNC_PATTERN_C = re.compile(
    r'^(?:(?:static|inline|extern|virtual|override|const|unsigned|signed|volatile|'
    r'void|int|char|float|double|long|short|bool|auto|size_t|uint\w+|int\w+)\s+)*'
    r'(?:\w+(?:::\w+)*\s+)*'
    r'(\w+(?:::\w+)*)\s*\(',
    re.MULTILINE
)
_FUNC_PATTERN_PY = re.compile(r'^((?:@\w+.*\n)*)\s*((?:async\s+)?def\s+\w+)', re.MULTILINE)
_CLASS_PATTERN_PY = re.compile(r'^(class\s+\w+)', re.MULTILINE)


def _chunk_code_file_regex(
    file_path: str, rel_path: str, content: str, lang_config: dict,
    max_chunk_size: int = 4000,
) -> List[Document]:
    """regex 기반 폴백 청킹: tree-sitter 미설치 시 사용."""
    lang_name = lang_config["name"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=200,
        separators=["\n\n\n", "\n\n", "\n", " ", ""],
    )
    texts = splitter.split_text(content)

    chunks: List[Document] = []
    for idx, text in enumerate(texts):
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

    imports: List[str] = []
    for line in content.split('\n')[:100]:
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
# CodeRAG
# ══════════════════════════════════════════════════════════════

class CodeRAG(BaseRAG):
    """FAISS + BM25 하이브리드 RAG (소스코드 전용).

    tree-sitter AST 기반 구조적 청킹으로 함수/클래스 경계를 정확히 보존한다.
    3단계 청크 체계(L1 파일 요약, L2 함수/클래스, L3 서브 청크)와
    5종 인덱스를 통해 심볼 정확 매칭 + 시맨틱 검색 + 서브 청크 재조립을 수행한다.
    """

    CODE_EXTENSIONS: Set[str] = set(_LANGUAGE_CONFIG.keys())

    def __init__(self, db_store_path: str = "./knowledge_base",
                 project_name: str = ""):
        code_model = CODE_EMBEDDING_MODEL or None
        super().__init__(db_store_path, embedding_model_override=code_model)

        self.project_name: str = project_name
        self.project_root: str = ""
        self.project_meta_path = os.path.join(self.db_store_path, "project_meta.pkl")

        self.symbol_index_path = os.path.join(self.db_store_path, "symbol_index.pkl")
        self.req_id_index_path = os.path.join(self.db_store_path, "req_id_index.pkl")
        self.file_path_index_path = os.path.join(self.db_store_path, "file_path_index.pkl")
        self.function_id_index_path = os.path.join(self.db_store_path, "function_id_index.pkl")
        self.file_manifest_path = os.path.join(self.db_store_path, "file_manifest.pkl")

        self.symbol_index: Dict[str, List[Document]] = {}
        self.req_id_index: Dict[str, List[Document]] = {}
        self.file_path_index: Dict[str, Document] = {}
        self.function_id_index: Dict[str, List[Document]] = {}
        self.file_manifest: Dict[str, float] = {}

    def is_db_exists(self) -> bool:
        """FAISS DB, BM25 캐시, 5종 인덱스가 모두 존재하는지 확인"""
        return (os.path.exists(self.faiss_path)
                and os.path.exists(self.bm25_path)
                and os.path.exists(self.symbol_index_path)
                and os.path.exists(self.file_path_index_path))

    def load_db(self):
        """기존 구축된 DB 로드 (pickle 인덱스를 병렬 로드하여 속도 향상)"""
        from concurrent.futures import ThreadPoolExecutor

        logger.info(f"📦 로컬에 구축된 CodeRAG DB를 '{self.db_store_path}'에서 로드 중...")
        self._load_vector_stores()

        if os.path.exists(self.project_meta_path):
            with open(self.project_meta_path, "rb") as f:
                meta = pickle.load(f)
                if not self.project_name:
                    self.project_name = meta.get("project_name", "")
                self.project_root = meta.get("project_root", "")

        # 5개 pickle 인덱스를 병렬 로드
        index_targets = [
            ("symbol_index", self.symbol_index_path),
            ("req_id_index", self.req_id_index_path),
            ("file_path_index", self.file_path_index_path),
            ("function_id_index", self.function_id_index_path),
            ("file_manifest", self.file_manifest_path),
        ]

        def _load_pickle(path: str):
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
            return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {attr: executor.submit(_load_pickle, path) for attr, path in index_targets}
            for attr, future in futures.items():
                result = future.result()
                if result is not None:
                    setattr(self, attr, result)

        self._setup_ensemble()
        logger.info(f"✅ CodeRAG DB 재로드 완료! "
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
        """소스코드 파일/폴더를 인덱싱하여 DB 구축."""
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

        project_root = source_path if os.path.isdir(source_path) else os.path.dirname(source_path)
        self.project_root = project_root

        if not self.project_name:
            self.project_name = os.path.basename(project_root)

        self._build_db_from_files(files, project_root)

    def _build_db_from_files(self, file_paths: List[str], project_root: str):
        """소스코드 파일 목록을 AST 기반 청킹 후 Vector DB와 인덱스 구축"""
        logger.info(f"🔨 {len(file_paths)}개의 소스 파일로 CodeRAG DB 구축 시작...")

        if _TREE_SITTER_AVAILABLE:
            supported = [ext for ext in sorted(self.CODE_EXTENSIONS) if ext in _TS_LANGUAGES]
            fallback = [ext for ext in sorted(self.CODE_EXTENSIONS) if ext not in _TS_LANGUAGES]
            logger.info(f"  🌳 tree-sitter 파싱: {', '.join(supported) if supported else '없음'}")
            if fallback:
                logger.info(f"  📝 regex 폴백: {', '.join(fallback)}")
        else:
            logger.warning("  ⚠️ tree-sitter 미설치 — 모든 파일에 regex 폴백 적용")

        all_splits: List[Document] = []

        for path in file_paths:
            if not os.path.exists(path):
                logger.warning(f"  ⚠️ 경고: '{path}' 파일을 찾을 수 없습니다.")
                continue

            ext = os.path.splitext(path)[1].lower()
            lang_config = _LANGUAGE_CONFIG.get(ext)
            if not lang_config:
                continue

            rel_path = os.path.relpath(path, project_root)
            logger.info(f"  📄 {rel_path} ({lang_config['name']})")

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"  ⚠️ 읽기 실패: {e}")
                continue

            source_bytes = content.encode('utf-8')

            parser = _get_ts_parser(ext)
            if parser is not None:
                tree = parser.parse(source_bytes)
                file_summary = _build_file_summary_chunk(
                    rel_path, source_bytes, tree.root_node, lang_config)
                all_splits.append(file_summary)
                code_chunks = _chunk_code_file_ts(
                    path, rel_path, source_bytes, lang_config, tree.root_node)
                all_splits.extend(code_chunks)
            else:
                file_summary = _build_file_summary_regex(rel_path, content, lang_config)
                all_splits.append(file_summary)
                code_chunks = _chunk_code_file_regex(path, rel_path, content, lang_config)
                all_splits.extend(code_chunks)

            self.file_manifest[path] = os.path.getmtime(path)

        if not all_splits:
            raise ValueError("소스코드에서 추출된 청크가 없습니다.")

        for doc in all_splits:
            doc.metadata["project"] = self.project_name

        self._save_vector_stores(all_splits, code_bm25_preprocessor)

        # ── 5종 인덱스 구축 ──
        self.symbol_index = {}
        self.req_id_index = {}
        self.file_path_index = {}
        self.function_id_index = {}

        for doc in all_splits:
            chunk_type = doc.metadata.get("chunk_type", "")

            if chunk_type == "file_summary":
                source = doc.metadata.get("source", "")
                self.file_path_index[source] = doc
                basename = os.path.basename(source)
                self.file_path_index[basename] = doc

            for sym in doc.metadata.get("symbols_defined", []):
                self.symbol_index.setdefault(sym, []).append(doc)

            for req_id in doc.metadata.get("requirement_ids", []):
                self.req_id_index.setdefault(req_id, []).append(doc)

            func_id = doc.metadata.get("function_id")
            if func_id:
                self.function_id_index.setdefault(func_id, []).append(doc)

        os.makedirs(self.db_store_path, exist_ok=True)

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
        logger.info(f"✅ CodeRAG DB 구축 완료!")
        logger.info(f"   청크: {len(all_splits)} | 심볼: {len(self.symbol_index)} | "
                     f"Req ID: {len(self.req_id_index)} | 파일: {len(self.file_path_index)}")
        logger.info(f"   저장 경로: '{self.db_store_path}'")

    TERM_INDEX_MAX_HITS = 10

    def _extract_code_terms(self, query: str) -> dict:
        """쿼리에서 검색 가능한 용어를 추출한다."""
        terms = {"file_paths": set(), "req_ids": set(), "symbols": set()}
        upper_query = query.upper()

        for word in re.findall(r'[\w./\\-]+\.\w{1,5}', query):
            ext = os.path.splitext(word)[1].lower()
            if ext in self.CODE_EXTENSIONS or ext == '.md':
                terms["file_paths"].add(word)

        terms["req_ids"].update(set(_REQ_ID_RE.findall(upper_query)) - _REQ_ID_EXCLUDE)

        for prefix, num in _CODE_REQ_PATTERN.findall(query):
            normalized = f"{prefix.upper()}-{num}"
            if normalized not in _REQ_ID_EXCLUDE:
                terms["req_ids"].add(normalized)

        for word in re.findall(r'\b([A-Za-z_]\w{2,})\b', query):
            if ('_' in word or
                re.search(r'[a-z][A-Z]', word) or
                word[0].isupper() and len(word) > 2):
                if word in self.symbol_index:
                    terms["symbols"].add(word)

        return terms

    def _reassemble_subchunks(self, docs: List[Document]) -> List[Document]:
        """검색 결과에서 서브 청크를 감지하여 완전한 함수로 재조립한다."""
        result: List[Document] = []
        reassembled_fids: Set[str] = set()

        for doc in docs:
            if not doc.metadata.get("is_subchunk"):
                result.append(doc)
                continue

            fid = doc.metadata.get("function_id", "")
            if fid in reassembled_fids:
                continue

            siblings = self.function_id_index.get(fid, [doc])
            siblings_sorted = sorted(
                siblings,
                key=lambda d: int(d.metadata.get("chunk_index", "1/1").split("/")[0])
            )

            merged_content = '\n'.join(s.page_content for s in siblings_sorted)
            merged_req_ids = sorted(set(
                rid for s in siblings_sorted
                for rid in s.metadata.get("requirement_ids", [])
            ))

            merged_meta = dict(siblings_sorted[0].metadata)
            merged_meta["is_subchunk"] = False
            merged_meta["chunk_index"] = None
            merged_meta["reassembled"] = True
            merged_meta["requirement_ids"] = merged_req_ids

            result.append(Document(page_content=merged_content, metadata=merged_meta))
            reassembled_fids.add(fid)

        return result

    def retrieve(self, query: str, top_k: int = 4) -> List[Document]:
        """코드 RAG 하이브리드 검색: 직접 매칭 + 앙상블 + 서브 청크 재조립."""
        if not self.ensemble_retriever:
            raise RuntimeError("검색할 DB가 로드되지 않았습니다. build_or_load()를 먼저 호출하세요.")

        terms = self._extract_code_terms(query)
        results: List[Document] = []
        seen: Set[str] = set()

        def _add_unique(doc: Document) -> None:
            key = doc.page_content[:200]
            if key not in seen:
                results.append(doc)
                seen.add(key)

        # ── Tier 1: 직접 검색 ──
        for fp in terms["file_paths"]:
            doc = self.file_path_index.get(fp)
            if doc:
                _add_unique(doc)

        for req_id in terms["req_ids"]:
            hits = self.req_id_index.get(req_id, [])
            if len(hits) <= self.TERM_INDEX_MAX_HITS:
                for doc in hits:
                    _add_unique(doc)

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
        reassembled = self._reassemble_subchunks(results)

        return reassembled[:top_k]

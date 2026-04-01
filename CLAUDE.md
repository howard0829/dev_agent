# DeepAssist 프로젝트 가이드

이 프로젝트는 Ollama(로컬 LLM), Gemini API, OpenRouter, Claude Agent SDK 기반 자율 코딩 에이전트 애플리케이션입니다.
코드를 수정하거나 기능을 추가할 때 반드시 따라야 하는 가이드라인입니다.

## 아키텍처 개요

Streamlit 웹 UI 프론트엔드와 `ClaudeAgentRunner` 백엔드가 결합된 구조입니다. 에이전트 모드에서는 `Todo List(Plan) 작성 -> 순차 실행(Execute) -> 검증` 워크플로우를 Claude SDK가 자율적으로 수행합니다.

### 모듈 구조

- **`models.py`**: 핵심 데이터 모델
  - `Task`, `Plan`, `ToolCallRecord` 데이터클래스
- **`llm_clients.py`**: LLM 클라이언트 (단순 채팅 모드용)
  - `BaseLLMClient` 추상 클래스, `OllamaClient`, `GeminiClient`, `OpenRouterClient`
  - `OpenRouterClient`: OpenAI 호환 API를 통해 Claude, GPT, Gemini 등 다양한 모델을 단일 키로 사용
- **`agent.py`**: Claude Agent SDK Runner
  - `ClaudeAgentRunner` 클래스: SDK 기반으로 구동되며 `on_status`/`on_tool_call` 콜백으로 UI와 연동
  - **LiteLLM 프록시 연동**: Ollama, Gemini, OpenRouter 선택 시, LiteLLM 로컬 프록시를 통해 Anthropic API 규격으로 변환
  - `ALLOWED_TOOLS`: SDK에 전달할 허용 도구 목록
  - **자동 룰 로더**: 작업 디렉토리에 `DeepAssist.md` 파일이 존재하면 시스템 프롬프트에 자동 주입
- **`app.py`**: 웹 프론트엔드 (Streamlit)
  - 두 개의 탭: **채팅** (에이전트 실행) + **워크스페이스** (파일 관리)
  - `st.session_state`를 활용한 상태 관리 (채팅 기록, 실행 로그 등)
  - 에이전트 모드와 단순 채팅 모드 지원
  - **LLM 프로바이더 4종**: Ollama, Gemini API, OpenRouter, Claude Agent SDK를 사이드바에서 선택
  - **파일 관리 탭**: `httpx`를 통해 FastAPI 서버(`server.py`)와 통신. **파일만 관리** (폴더 기능 없음)
- **`mcp_server.py`**: MCP(Model Context Protocol) 서버
  - `FastMCP`를 사용하여 작성, `rag.py`를 감싸서 RAG 도구를 제공
  - **MCP 도구 3종**:
    - `list_knowledge_dbs`: 구축된 전체 DB 목록 조회 (유형/이름/경로)
    - `search_knowledge`: 하이브리드 검색 (db_path 미지정 시 쿼리 기반 DB 자동 선택)
    - `search_web_and_scrape`: DuckDuckGo + BeautifulSoup 기반 웹 검색/스크래핑
  - **쿼리 기반 DB 자동 선택** (`_match_dbs_by_query`):
    - 쿼리에서 DB명/프로젝트명 키워드를 독립 토큰 매칭으로 감지
    - "uni92K 코드 만들어줘" → uni92K DB만 선택 (uni93K 제외)
    - "nvme에서 telemetry" → NVMe 관련 DB만 선택
    - 매칭 없으면 전체 DB Fan-out 검색으로 폴백
    - `_keyword_in_query`: ASCII 영숫자 경계 검사로 서브스트링 오매칭 방지
  - **소스 유형 자동 감지** (`_detect_source_type`, `_detect_rag_type`): 파일 확장자/DB 인덱스로 MarkdownRAG vs CodeRAG 자동 판별
  - 새로운 도구를 추가하려면 `@mcp.tool()` 함수를 추가
- **`rag.py`**: FAISS + BM25 하이브리드 RAG 모듈 (마크다운 + 소스코드 통합)
  - **아키텍처**: `BaseRAG` → `MarkdownRAG` / `CodeRAG` 상속 구조
  - 임베딩 프로바이더(Ollama/Gemini) 및 모델은 `.env` 파일로 관리
  - **하이브리드 검색**: FAISS(의미론적, 60%) + BM25(키워드, 40%)를 `EnsembleRetriever`로 결합
  - **DB 재사용**: `build_or_load(source_path)` — DB 있으면 즉시 로드, 없으면 새로 구축
  - **MarkdownRAG** — 마크다운 기술문서 전용:
    - `doc_name` 파라미터: 문서 식별명 (미지정 시 파일명에서 자동 추출), `doc_meta.pkl`로 DB에 저장
    - 프론트매터 스킵 (`_find_content_start`): 목차(TOC) 자동 감지 스킵
    - 구조적 청킹 (`_split_md_by_header_boundary`): 헤더 경계 1차 분할 + 테이블 행 우선 2차 분할
    - 2차 분할 시 구조적 문맥 주입 (`_extract_chunk_context`): 접두사 + 마크다운 헤더 + 테이블 헤더행+구분선
    - 용어 인덱스 (`term_index`): Req ID + 약어 자동 추출, 희귀 용어 직접 검색 + 앙상블 보충
    - BM25 전처리기: Req ID 하이픈 보존 (`TEL-6` → `tel-6`)
    - DB 구성: `faiss_index/` + `bm25_retriever.pkl` + `term_index.pkl` + `doc_meta.pkl`
  - **CodeRAG** — 소스코드 전용 (Python/C/C++/Java):
    - `project_name` 파라미터: 프로젝트 식별명 (미지정 시 폴더명에서 자동 추출), `project_meta.pkl`로 DB에 저장
    - tree-sitter AST 기반 청킹 (미설치 시 regex 폴백)
    - Python 데코레이터 인식 (`decorated_definition`): `@test_case("TEL-2")` 등이 함수와 함께 청킹
    - 3단계 청크 체계: L1 파일 요약, L2 함수/클래스, L3 서브 청크
    - 서브 청크 재조립 (`_reassemble_subchunks`): `function_id`로 연결된 서브 청크를 합쳐 완전한 함수 반환
    - 코드 내 Req ID 추출: 하이픈(`TEL-2`) + 언더스코어(`TEL_2` → `TEL-2`) 정규화
    - 5종 인덱스: `symbol_index`, `req_id_index`, `file_path_index`, `function_id_index`, `file_manifest`
    - 코드 BM25 전처리기: CamelCase/snake_case 분리 + Req ID 하이픈 보존
    - DB 구성: `faiss_index/` + `bm25_retriever.pkl` + `symbol_index.pkl` + `req_id_index.pkl` + `file_path_index.pkl` + `function_id_index.pkl` + `file_manifest.pkl` + `project_meta.pkl`
    - 언어 확장: `_LANGUAGE_CONFIG` 딕셔너리에 한 줄 추가로 새 언어 지원
- **`server.py`**: FastAPI 파일/워크스페이스 관리 서버
  - 클라이언트 IP + User-Agent 기반 세션 ID 생성, `workspaces/{session_id}/` 폴더 자동 할당
  - **주요 설정 상수** (파일 상단):
    - `MAX_FILE_SIZE_MB = 100`, `MAX_WORKSPACE_SIZE_MB = 100`
    - `WORKSPACE_EXPIRE_HOURS = 24`, `ALLOWED_EXTENSIONS`
  - **API 엔드포인트**: `/api/session`, `/api/files/list`, `/api/files/listdir`, `/api/files/upload`, `/api/files/download/{filename}`, `/api/files/{filename}` (DELETE), `/api/files/read/{file_path}`, `/api/files/write`, `/api/health`
  - Path Traversal 공격 방지 로직 내장 (`is_safe_path()`)
  - 비활성 워크스페이스를 30분 주기로 자동 정리
  - 평면(flat) 구조: 폴더 생성/삭제 API 없음

## 주요 제약 조건

1. **대형 파일 처리**: `read_file` 도구는 500줄 이상의 파일을 한 번에 반환하지 않음. `start_line`/`end_line`으로 500줄 단위 청크로 읽어야 함
2. **파일 수정**: Search & Replace 방식. `old_text`는 파일 내에서 유일해야 작동
3. **상태 관리**: `app.py` 수정 시 Streamlit 재실행 특성을 고려하여 `st.session_state`를 안전하게 다룰 것
4. **RAG 환경 설정**: `.env`에서 `EMBEDDING_PROVIDER`, `OLLAMA_EMBEDDING_MODEL`, `GEMINI_EMBEDDING_MODEL` 설정 필수
6. **OpenRouter**: openrouter.ai에서 API Key 발급 후 `.env`의 `OPENROUTER_API_KEY` 또는 사이드바에서 입력. Anthropic API Key 없이도 Claude 모델 사용 가능
5. **파일 관리 정책**: 워크스페이스는 파일만 관리 (폴더 계층 구조 미지원). 단일 파일 및 전체 워크스페이스 용량 제한 각 100MB. 24시간 비활성 시 자동 삭제

## 코드 컨벤션

### 기본 스타일
- Python 3, `typing` 모듈을 통한 Type Hinting 적극 사용
- 4-스페이스 들여쓰기, PEP 8 준수
- 모듈 상단, 클래스 선언부, 복잡한 함수에 docstring 포함
- 단일 책임 원칙 적용: 모델(`models.py`), 클라이언트(`llm_clients.py`), 에이전트(`agent.py`), 도구(`mcp_server.py`), RAG(`rag.py`), UI(`app.py`), 서버(`server.py`)

### 언어 규칙
- **Docstring, 주석, UI 메시지, 프롬프트**: 한국어
- **변수명, 함수명, 클래스명**: 영어 (PEP 8 네이밍)
- 콘솔 출력은 이모지 접두사 + 한국어 메시지: `print(f"🔨 {count}개의 파일로 DB 구축 시작...")`

### 네이밍
- 클래스: `PascalCase` (예: `ClaudeAgentRunner`, `MarkdownRAG`)
- 함수/변수: `snake_case`, 비공개 함수는 `_` 접두사 (예: `_short_path()`, `_parse_markdown_todos()`)
- 상수: 모듈 최상단에 `UPPER_SNAKE_CASE` 선언 (예: `ALLOWED_TOOLS`, `MAX_FILE_SIZE_MB`)

### Import 순서
1. 표준 라이브러리 (`os`, `json`, `re`, `time`, …)
2. 서드파티 라이브러리 (`requests`, `streamlit`, `fastapi`, …)
3. 프로젝트 내부 모듈 (`from models import …`, `from llm_clients import …`)
4. 선택적 임포트는 `try-except`로 감싸고 가용 플래그 설정 (`CLAUDE_SDK_AVAILABLE = True/False`)

### 문자열 포매팅
- **f-string 전용** 사용. `.format()`, `%` 포매팅 사용 금지

### 에러 처리
- `try-except`로 에러 핸들링하여 앱/에이전트 루프 중단 방지
- 예외는 구체적 → 범용 순서로 처리:
  ```python
  except requests.Timeout:
      return {"error": "API 호출 타임아웃 (300초)"}
  except requests.ConnectionError:
      return {"error": "API 연결 끊김"}
  except Exception as e:
      return {"error": f"API 오류: {e}"}
  ```
- LLM 클라이언트 반환 형식 통일:
  - 성공: `{"message": {"role": "assistant", "content": "...", "tool_calls": [...]}}`
  - 실패: `{"error": "에러 메시지"}`
- 연결 체크: `tuple[bool, str]` 반환 (예: `(True, "연결됨 (모델: ...)")`)
- MCP 도구 에러: 문자열로 반환 (예: `"❌ 검색 중 오류: ..."`)

### 경로 처리
- 파일 경로는 `os.path.expanduser` → `os.path.abspath` → `os.path.join` 순서로 처리하여 항상 절대경로 사용

## 기능 추가 가이드

**새로운 도구(Tool) 추가:**
1. `mcp_server.py`에 `@mcp.tool()` 데코레이터와 함께 함수 작성
2. 타입 힌트와 상세 Docstring으로 Claude Agent가 자동 인식
3. 에러 시 에러 메시지 텍스트를 리턴하여 에이전트가 우회할 수 있도록 설계

**LLM 프로바이더 추가:**
1. `llm_clients.py`의 `BaseLLMClient` 추상 클래스 상속
2. `check_connection`, `chat` 메서드(스트리밍 포함) 구현
3. `app.py`에 프로바이더 선택 UI 추가 및 클라이언트 초기화 로직 등록

**RAG(`rag.py`) 수정:**
- `MarkdownRAG`의 공개 메서드(`build_or_load`, `retrieve`) 인터페이스 유지
- `bm25_preprocessor` 함수는 반드시 모듈 최상단(글로벌 스코프)에 선언 (pickle 직렬화 요구)
- 마크다운(.md) 파일만 지원. PDF 관련 코드는 의도적으로 제거됨
- 새 문서 추가: `__main__`의 `DOCS` 리스트에 항목 추가 (`label`, `source`, `db_path`, `queries`)
- 임베딩 모델 변경 시 해당 knowledge 폴더 삭제 후 DB 재구축 필요
- 청킹 파라미터: `min_chunk_size=1000`, `max_chunk_size=3000`
- 주요 정규식 패턴 (모듈 상단):
  - `_REQ_ID_RE`: Requirement ID (`TEL-6`, `SEC-3`), `_REQ_ID_EXCLUDE`로 오탐 방지
  - `_ABBR_DEF_RE`: 약어 정의 (`Full Name (ABBR)`)
  - `_TABLE_ROW_RE` / `_TABLE_SEP_RE`: 테이블 행/구분선
- 검색 분기: `TERM_INDEX_MAX_HITS`(10) 이하 매핑인 희귀 용어만 직접 검색, 초과 시 앙상블
- DB는 3개 파일로 구성: `faiss_index/`, `bm25_retriever.pkl`, `term_index.pkl` — 하나라도 없으면 재구축

**파일 서버(`server.py`) 수정:**
- `is_safe_path()` 보안 검증 함수 우회/제거 금지
- 상수는 파일 상단에서 관리
- FastAPI 서버는 파일 관리만 담당. 에이전트 실행 로직은 `app.py`에서 처리
- 폴더 생성/삭제 API 없음. 평면 구조로 파일만 관리

**워크스페이스 탭(`app.py`) 수정:**
- `st.session_state` 키: `editing_file`, `_uploaded_hash`
- 파일 목록은 `/api/files/listdir?path=` API로 조회, `type == "file"` 항목만 표시
- 폴더 탐색 기능은 제거됨. 추가하지 말 것

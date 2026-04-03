# 프로젝트 가이드

멀티 앱 아키텍처 기반의 자율 코딩 에이전트 플랫폼입니다.
Ollama(로컬 LLM), Gemini API, vLLM, OpenAI 호환 API를 지원하며, 여러 앱(DeepAssist, TestMancer 등)을 하나의 플랫폼에서 전환하여 사용할 수 있습니다.
코드를 수정하거나 기능을 추가할 때 반드시 따라야 하는 가이드라인입니다.

## 아키텍처 개요

**멀티 앱 런처(`app.py`)** + **앱별 페이지 모듈(`apps/{name}/page.py`)** + **공유 유틸리티(`core/`)** 구조입니다. 메인 화면 상단 카드 바에서 앱을 선택하면 해당 앱의 `page.py`가 사이드바와 메인 화면을 **독립적으로** 렌더링합니다. `core/` 모듈은 강제가 아닌 유틸리티 라이브러리로, 앱이 선택적으로 import하여 사용합니다.

### 모듈 구조

- **`app.py`**: 멀티 앱 런처 (Streamlit)
  - `apps/` 레지스트리에서 앱을 자동 검색하여 상단 카드 바에 표시
  - 카드 클릭으로 앱 전환, `st.session_state["_selected_app_id"]`로 선택 상태 관리
  - 선택된 앱의 `page.py`에 사이드바/메인 렌더링을 **완전 위임**
  - 런처 자체는 앱 스위처 + 스타일 적용만 담당
- **`apps/`**: 앱 레지스트리 + 앱별 독립 모듈
  - **`apps/__init__.py`**: `discover_apps()` — `apps/` 하위 패키지에서 `config.py` + `page.py`를 자동 검색. `page.py`에 `init_app_session`, `render_sidebar`, `render_main` 3개 함수 필수
  - **`apps/deep_assist/`**: DeepAssist (자율 코딩 에이전트)
    - `config.py`: `APP_CONFIG` dict (`id`, `name`, `icon`, `description`, `tabs`, `default_mode`, `custom_css`)
    - `page.py`: `core/` 유틸리티를 재사용. 사이드바(LLM 4종 프로바이더: Ollama/Gemini/vLLM/OpenAI), 메인(채팅+워크스페이스 탭)
  - **`apps/test_mancer/`**: TestMancer (테스트 자동화 에이전트)
    - `config.py`: TestMancer 전용 설정
    - `page.py`: **자체 구현**. 사이드바(LLM 설정 + 테스트 프레임워크/대상 설정), 메인(채팅+테스트 결과 탭), 전용 세션(`test_results`, `test_framework`, `test_target_dir`), 테스트 컨텍스트 프롬프트 주입
- **`core/`**: 공유 유틸리티 (강제 아님, 앱이 선택적으로 import)
  - **`core/session.py`**: 네임스페이스 세션 상태 관리. `ns(prefix, key)` → `"{prefix}.{key}"` 형식으로 앱별 세션 격리. `init_session(prefix)`, `get_state(prefix, key)`, `set_state(prefix, key, value)`, `reset_chat(prefix)`, `reset_logs(prefix)`, `display_path()`
  - **`core/styles.py`**: 공통 CSS (앱 스위처 카드, 채팅, 사이드바 등). `apply_common_styles()`, `apply_custom_css(css)`
  - **`core/sidebar.py`**: LLM 프로바이더 선택 UI. `render_llm_sidebar()` → 설정 dict 반환 (`llm_provider`, `model_name`, `api_key`, `ollama_url`, `vllm_url`, `enable_thinking`, `agent_mode`, `backend_mode`). vLLM/OpenAI 이외 프로바이더에서 백엔드 모드 선택 UI(`auto`/`proxy`/`native`) 표시. OpenAI 선택 시 `.env`의 `ANTHROPIC_*` 환경변수에서 자동 로드. 앱이 사이드바를 자체 구현할 경우 사용하지 않아도 됨
  - **`core/chat_ui.py`**: 채팅 탭 공통 로직. `render_chat_tab(prefix, provider_cfg)`. 에이전트 모드에서 `progress_container`를 통해 메인 채팅 영역에 계획/Task 진행/도구 호출을 실시간 표시. 앱이 채팅 UI를 자체 구현할 경우 사용하지 않아도 됨
  - **`core/workspace_ui.py`**: 워크스페이스 파일 관리 탭. `render_workspace_tab(prefix)`. `httpx`를 통해 FastAPI 서버(`server.py`)와 통신
- **`config.py`**: 중앙 설정 모듈 (환경변수 기반)
  - 모든 URL, 포트, 상수를 환경변수에서 로드. `.env` 파일 자동 로드
  - LLM 기본값: `OLLAMA_DEFAULT_URL`, `VLLM_DEFAULT_URL`, `OLLAMA_DEFAULT_MODEL`, `VLLM_DEFAULT_MODEL`, `GEMINI_DEFAULT_MODEL`
  - OpenAI Direct: `OPENAI_DIRECT_BASE_URL`(`ANTHROPIC_BASE_URL`), `OPENAI_DIRECT_API_KEY`(`ANTHROPIC_API_KEY`), `OPENAI_DIRECT_MODEL`(`ANTHROPIC_DEFAULT_SONNET_MODEL`)
  - 임베딩: `EMBEDDING_PROVIDER`, `OLLAMA_EMBEDDING_MODEL`, `GEMINI_EMBEDDING_MODEL`, `GEMINI_API_KEY`, `CODE_EMBEDDING_MODEL`
  - 프록시: `PROXY_PORT`, `PROXY_MAX_WAIT`
  - 서버: `FILE_SERVER_URL`, `FILE_SERVER_PORT`, `WORKSPACES_ROOT`, `MAX_FILE_SIZE_MB`, `MAX_WORKSPACE_SIZE_MB`, `WORKSPACE_EXPIRE_HOURS`, `CLEANUP_INTERVAL_MINUTES`, `ALLOWED_EXTENSIONS`
  - CORS: `CORS_ORIGINS`
  - 에이전트: `AGENT_MAX_TURNS`, `DEEPASSIST_MD_MAX_SIZE`
  - Knowledge: `KNOWLEDGE_BASE_DIR`
- **`models.py`**: 핵심 데이터 모델
  - `Task`, `Plan`, `ToolCallRecord` 데이터클래스 (`to_dict()` 직렬화 메서드 포함)
  - `ProviderConfig` TypedDict: LLM 프로바이더 설정 딕셔너리 타입
  - `CallbackSet` TypedDict: 콜백 함수 세트 타입
- **`llm_clients.py`**: LLM 클라이언트 (단순 채팅 모드용)
  - `BaseLLMClient` 추상 클래스, `OllamaClient`, `GeminiClient`
- **`backend_strategy.py`**: 백엔드 전략 모듈 (Strategy 패턴)
  - `BackendStrategy` 추상 클래스: `check()`, `activate(event_queue)`, `cleanup(event_queue)` 인터페이스
  - **`ProxyStrategy`**: claude-code-proxy 경유. Ollama/Gemini → Claude API 형식 변환. Claude 특수 파라미터(context_management 등) 처리와 tool calling 변환을 프록시가 자체 처리
  - **`VllmStrategy`**: ProxyStrategy 상속. vLLM 서빙 엔진(OpenAI 호환 API) + claude-code-proxy 경유. Guided Decoding(Outlines)으로 tool call JSON 100% 유효성 보장
  - **`OllamaNativeStrategy`**: Ollama v0.14+ 네이티브 Anthropic API 호환 모드. 프록시 없이 직접 연결. `count_tokens` 이슈 우회를 위해 3개 모델 환경변수 매핑
  - **`OpenAIDirectStrategy`**: OpenAI 호환 엔드포인트에 직접 연결. 프록시 프로세스를 자체 기동하지 않고 `.env`의 `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_DEFAULT_*_MODEL` 환경변수만 설정하여 Claude Agent SDK가 외부 프록시/API에 연결
  - `select_strategy()`: 프로바이더 + 백엔드 모드에 따라 전략 자동 선택. `auto` 모드에서는 Native 시도 후 실패 시 Proxy 폴백. vLLM/OpenAI는 전용 전략 자동 선택
  - 공유 유틸리티: `_check_ollama()`, `_read_log_tail()`, `_detect_dropped_params()`, `_safe_unlink()`
- **`agent.py`**: Claude Agent SDK Runner
  - `ClaudeAgentRunner` 클래스: SDK 기반으로 구동되며 `on_status`/`on_tool_call`/`on_agent_text` 콜백으로 UI와 연동
  - **실시간 진행 표시**: `on_agent_text` 콜백으로 에이전트의 계획/설명 텍스트를 메인 채팅 영역에 실시간 스트리밍. `"agent_text"` 이벤트로 도구 호출 로그(`"status"`)와 분리
  - **강화된 시스템 프롬프트**: `forced_prompt`가 각 Task 시작 시 `🔄 Task N 시작:` 마커 출력, 도구 호출 전 이유 설명을 지시하여 진행 상황의 가독성 향상
  - **미완료 Task 자동 재시도**: `_async_run()` 완료 후 미완료(pending/in_progress) Task를 자동 감지하여 에이전트에게 재프롬프트. `MAX_CONTINUATIONS`(기본 2회)까지 재시도. 최종 완료 상태를 `✅ 모든 Task 완료` 또는 `⚠️ 미완료 Task N개` 형태로 보고
  - **Task 상태 추적**: 모델의 명시적 마커(`🔄 Task N 시작`, `✅ Task N 완료`) 기반으로 정확하게 추적. 도구 호출 횟수 기반 자동 완료는 제거
  - **백엔드 전략 위임**: `backend_mode` 파라미터(`auto`/`proxy`/`native`)로 `backend_strategy.py`의 전략을 선택. `_async_run()`에서 `strategy.activate()` → SDK 호출 → `strategy.cleanup()` 순서로 실행
  - `ALLOWED_TOOLS`: SDK에 전달할 허용 도구 목록
  - **자동 룰 로더**: 작업 디렉토리에 `DeepAssist.md` 파일이 존재하면 시스템 프롬프트에 자동 주입 (`DEEPASSIST_MD_MAX_SIZE` 크기 제한 적용)
  - `threading.Event` 기반 이벤트 루프 (Queue 폴링 대신 효율적 동기화)
- **`mcp_server.py`**: MCP(Model Context Protocol) 서버
  - `FastMCP`를 사용하여 작성, `rag/` 패키지를 감싸서 RAG 도구를 제공
  - **RAG 인스턴스 캐시**: `_rag_cache` 딕셔너리로 DB별 RAG 인스턴스 재사용
  - **안전한 pickle 로드**: `_safe_pickle_load()` 함수로 역직렬화 오류 안전 처리
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
- **`rag/`**: FAISS + BM25 하이브리드 RAG 패키지 (마크다운 + 소스코드 통합)
  - **`rag/__init__.py`**: 기존 import 호환 (`from rag import MarkdownRAG, CodeRAG, BaseRAG`). pickle 호환을 위해 `bm25_preprocessor`, `code_bm25_preprocessor` re-export
  - **`rag/constants.py`**: 공통 상수, 정규식 패턴 (`_REQ_ID_RE`, `_HEADER_RE`, `_TABLE_ROW_RE` 등), `_LANGUAGE_CONFIG` tree-sitter 언어 레지스트리, 그래머 캐시 (`_TS_LANGUAGES`, `_TS_PARSERS`)
  - **`rag/utils.py`**: `bm25_preprocessor` (마크다운 BM25 전처리), `_extract_chunk_context` (2차 분할 시 구조적 문맥 추출)
  - **`rag/base.py`**: `BaseRAG` 클래스 — 임베딩 초기화 (`config.py`에서 설정 로드, `load_dotenv()` 제거), FAISS/BM25 저장/로드, 앙상블 검색. FAISS 배치 빌드 (10,000개 단위, 대규모 문서셋 메모리 효율 향상)
  - **`rag/markdown.py`**: `MarkdownRAG` + 마크다운 청킹 유틸리티 (`_find_content_start`, `_split_md_by_header_boundary`). 용어 인덱스, 희귀 용어 직접 검색 + 앙상블 보충
  - **`rag/code.py`**: `CodeRAG` + `code_bm25_preprocessor` + tree-sitter AST 헬퍼 + regex 폴백. 5종 인덱스, 서브 청크 재조립. `ThreadPoolExecutor`로 5개 pickle 인덱스 병렬 로드
  - **`rag/__main__.py`**: 테스트 유틸리티 (`python -m rag [markdown|code|all]`)
  - **아키텍처**: `BaseRAG` → `MarkdownRAG` / `CodeRAG` 상속 구조
  - 임베딩 프로바이더(Ollama/Gemini) 및 모델은 `config.py`를 통해 관리
  - **하이브리드 검색**: FAISS(의미론적, 60%) + BM25(키워드, 40%)를 `EnsembleRetriever`로 결합
  - **DB 재사용**: `build_or_load(source_path)` — DB 있으면 즉시 로드, 없으면 새로 구축
  - 언어 확장: `_LANGUAGE_CONFIG` 딕셔너리에 한 줄 추가로 새 언어 지원
- **`server.py`**: FastAPI 파일/워크스페이스 관리 서버
  - 클라이언트 IP + User-Agent 기반 세션 ID 생성, `workspaces/{session_id}/` 폴더 자동 할당
  - 모든 설정은 `config.py`에서 로드 (`MAX_FILE_SIZE_MB`, `MAX_WORKSPACE_SIZE_MB`, `WORKSPACE_EXPIRE_HOURS`, `ALLOWED_EXTENSIONS`, `CORS_ORIGINS` 등)
  - **API 엔드포인트**: `/api/session`, `/api/files/list`, `/api/files/listdir`, `/api/files/upload`, `/api/files/download/{filename}`, `/api/files/{filename}` (DELETE), `/api/files/read/{file_path}`, `/api/files/write`, `/api/health`
  - Path Traversal 공격 방지 로직 내장 (`is_safe_path()` — symlink 감지 포함)
  - 비활성 워크스페이스를 30분 주기로 자동 정리
  - 구조화 로깅 (`logging` 모듈 사용)
  - 평면(flat) 구조: 폴더 생성/삭제 API 없음
- **`tests/`**: pytest 테스트 스위트
  - `conftest.py`: 공통 fixture (`tmp_workspace`, `mock_session_state`)
  - `test_config.py`: config.py 환경변수 로딩 테스트
  - `test_models.py`: Task/Plan/ToolCallRecord `to_dict()` 직렬화 테스트
  - `test_session.py`: core/session.py 네임스페이스 세션 상태 테스트
  - `test_server.py`: FastAPI 엔드포인트 테스트 (TestClient)
  - `test_rag_utils.py`: BM25 전처리, 청크 문맥 추출, Req ID regex 테스트

## 주요 제약 조건

1. **대형 파일 처리**: `read_file` 도구는 500줄 이상의 파일을 한 번에 반환하지 않음. `start_line`/`end_line`으로 500줄 단위 청크로 읽어야 함
2. **파일 수정**: Search & Replace 방식. `old_text`는 파일 내에서 유일해야 작동
3. **상태 관리**: Streamlit 재실행 특성을 고려하여 `st.session_state`를 안전하게 다룰 것. 모든 세션 키는 `core/session.py`의 `ns(prefix, key)` → `"{prefix}.{key}"` 형식 네임스페이스를 사용하여 앱 간 충돌을 방지
4. **RAG 환경 설정**: `config.py`가 `.env`에서 `EMBEDDING_PROVIDER`, `OLLAMA_EMBEDDING_MODEL`, `GEMINI_EMBEDDING_MODEL` 등을 자동 로드. 새 설정 추가 시 `config.py`에 상수를 정의하고 다른 모듈에서 import하여 사용
5. **파일 관리 정책**: 워크스페이스는 파일만 관리 (폴더 계층 구조 미지원). 단일 파일 및 전체 워크스페이스 용량 제한 각 100MB. 24시간 비활성 시 자동 삭제

## 코드 컨벤션

### 기본 스타일
- Python 3, `typing` 모듈을 통한 Type Hinting 적극 사용
- 4-스페이스 들여쓰기, PEP 8 준수
- 모듈 상단, 클래스 선언부, 복잡한 함수에 docstring 포함
- 단일 책임 원칙 적용: 런처(`app.py`), 공유 코어(`core/`), 앱 설정(`apps/`), 설정(`config.py`), 모델(`models.py`), 클라이언트(`llm_clients.py`), 에이전트(`agent.py`), 도구(`mcp_server.py`), RAG(`rag/`), 서버(`server.py`), 테스트(`tests/`)

### 언어 규칙
- **Docstring, 주석, UI 메시지, 프롬프트**: 한국어
- **변수명, 함수명, 클래스명**: 영어 (PEP 8 네이밍)
- 로깅은 `logging` 모듈 사용. 각 모듈에 `logger = logging.getLogger(__name__)` 선언
  - `logger.info()`: 정상 동작 기록, `logger.warning()`: 비정상 상황, `logger.error()`: 오류
  - Streamlit UI 스트리밍 출력(`stream_to_terminal=True`)에서만 `print()` 허용

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

**백엔드 전략(`backend_strategy.py`) 수정/추가:**
- `BackendStrategy` 추상 클래스의 3개 메서드(`check`, `activate`, `cleanup`) 인터페이스 유지
- 새 백엔드 추가: `BackendStrategy`를 상속하는 새 클래스 작성 → `select_strategy()`에 분기 추가
- `ProxyStrategy`: claude-code-proxy를 사용. 프로바이더별 환경변수(`PREFERRED_PROVIDER`, `BIG_MODEL`, `SMALL_MODEL` 등)를 설정하여 프록시가 라우팅 수행
- `VllmStrategy`: `ProxyStrategy`를 상속. vLLM의 OpenAI 호환 API를 `OPENAI_BASE_URL`로 지정하여 claude-code-proxy 경유
- `OllamaNativeStrategy`: Ollama Anthropic 호환 모드 관련 이슈는 `check()`에서 `/v1/messages` 엔드포인트 접근 가능 여부로 사전 검증
- `OpenAIDirectStrategy`: 프록시 미기동. `.env`의 `ANTHROPIC_*` 환경변수를 설정하여 외부 프록시/API에 직접 연결. `check()`에서 `/health` 엔드포인트 확인 (실패해도 연결 시도)
- `select_strategy()`의 auto 모드 폴백 로직: Native → Proxy 순서. vLLM/OpenAI는 전용 전략 자동 선택

**새로운 도구(Tool) 추가:**
1. `mcp_server.py`에 `@mcp.tool()` 데코레이터와 함께 함수 작성
2. 타입 힌트와 상세 Docstring으로 Claude Agent가 자동 인식
3. 에러 시 에러 메시지 텍스트를 리턴하여 에이전트가 우회할 수 있도록 설계

**LLM 프로바이더 추가 (core/ 사용 앱):**
1. `llm_clients.py`의 `BaseLLMClient` 추상 클래스 상속
2. `check_connection`, `chat` 메서드(스트리밍 포함) 구현
3. `core/sidebar.py`의 `render_llm_sidebar()`에 프로바이더 선택 UI 추가
4. `core/chat_ui.py`의 `_get_agent()`에 클라이언트 초기화 로직 등록
5. 자체 사이드바를 가진 앱은 해당 앱의 `page.py`에서 직접 구현

**새 앱 추가:**
1. `apps/{app_name}/` 디렉토리 생성
2. `apps/{app_name}/__init__.py` 작성
3. `apps/{app_name}/config.py`에 `APP_CONFIG` 딕셔너리 정의 (`id`, `name`, `icon`, `description`, `tabs`, `default_mode`, `custom_css`)
4. `apps/{app_name}/page.py`에 3개 필수 함수 구현:
   - `init_app_session(prefix)`: 앱 전용 세션 상태 초기화
   - `render_sidebar(prefix) -> dict`: 앱 전용 사이드바 렌더링, 설정 dict 반환
   - `render_main(prefix, sidebar_cfg)`: 앱 전용 메인 화면 렌더링
5. `core/` 유틸리티 재사용 가능 (선택사항). 완전 자체 구현도 가능
6. 재시작 시 `discover_apps()`가 `config.py` + `page.py`를 자동 검색하여 등록

**RAG(`rag/` 패키지) 수정:**
- `MarkdownRAG`/`CodeRAG`의 공개 메서드(`build_or_load`, `retrieve`) 인터페이스 유지
- `bm25_preprocessor` 함수는 `rag/utils.py`에 정의, `rag/__init__.py`에서 re-export (pickle 호환 필수)
- `code_bm25_preprocessor` 함수는 `rag/code.py`에 정의, `rag/__init__.py`에서 re-export (pickle 호환 필수)
- 마크다운(.md) 파일만 지원. PDF 관련 코드는 의도적으로 제거됨
- 새 문서 추가: `rag/__main__.py`의 `TESTS` 리스트에 항목 추가 (`label`, `source`, `db_path`, `queries`)
- 임베딩 모델 변경 시 해당 knowledge 폴더 삭제 후 DB 재구축 필요
- 임베딩 설정은 `config.py`에서 로드 (`EMBEDDING_PROVIDER`, `OLLAMA_EMBEDDING_MODEL` 등). `load_dotenv()` 직접 호출 금지
- 청킹 파라미터: `min_chunk_size=1000`, `max_chunk_size=3000`
- 주요 정규식 패턴 (`rag/constants.py`):
  - `_REQ_ID_RE`: Requirement ID (`TEL-6`, `SEC-3`), `_REQ_ID_EXCLUDE`로 오탐 방지
  - `_ABBR_DEF_RE`: 약어 정의 (`Full Name (ABBR)`)
  - `_TABLE_ROW_RE` / `_TABLE_SEP_RE`: 테이블 행/구분선
- 검색 분기: `TERM_INDEX_MAX_HITS`(10) 이하 매핑인 희귀 용어만 직접 검색, 초과 시 앙상블
- MarkdownRAG DB는 3개 파일로 구성: `faiss_index/`, `bm25_retriever.pkl`, `term_index.pkl` — 하나라도 없으면 재구축
- FAISS 배치 빌드: `BaseRAG.FAISS_BATCH_SIZE`(10,000) 단위로 인덱싱하여 대규모 문서셋의 메모리 효율 향상
- CodeRAG 병렬 로드: `ThreadPoolExecutor`로 5개 pickle 인덱스를 병렬 로드하여 DB 로드 시간 단축

**파일 서버(`server.py`) 수정:**
- `is_safe_path()` 보안 검증 함수 우회/제거 금지 (symlink 감지 포함)
- 상수는 `config.py`에서 import하여 사용. 하드코딩 금지
- FastAPI 서버는 파일 관리만 담당. 에이전트 실행 로직은 `app.py`에서 처리
- 폴더 생성/삭제 API 없음. 평면 구조로 파일만 관리

**테스트(`tests/`) 수정:**
- 새 기능 추가 시 `tests/` 디렉토리에 대응하는 테스트 파일 추가 권장
- `pytest tests/ -v`로 전체 테스트 실행
- `conftest.py`의 `tmp_workspace`, `mock_session_state` fixture 활용
- Streamlit `st.session_state` 의존 코드는 `mock_session_state` fixture로 모킹

**워크스페이스 탭(`core/workspace_ui.py`) 수정:**
- 네임스페이스 세션 키: `{prefix}.editing_file`, `{prefix}._uploaded_hash`
- 모든 Streamlit 위젯 `key`에 `{prefix}_` 접두사 필수 (앱 간 위젯 충돌 방지)
- 파일 목록은 `/api/files/listdir?path=` API로 조회, `type == "file"` 항목만 표시
- 폴더 탐색 기능은 제거됨. 추가하지 말 것

**채팅 탭(`core/chat_ui.py`) 수정:**
- `core/chat_ui.py`를 사용하는 앱(DeepAssist 등)에만 해당
- 네임스페이스 세션 키: `{prefix}.messages`, `{prefix}.tool_log`, `{prefix}.is_running` 등
- 콜백 팩토리 `_make_callbacks(prefix, ...)`: todo/status 플레이스홀더와 prefix를 바인딩
- 에이전트 생성 `_get_agent(prefix, provider_cfg, callbacks, is_agent_mode)`: provider_cfg dict에서 설정 읽음
- **실시간 진행 표시**: `progress_container` + `progress_state` dict로 에이전트 텍스트(계획/설명)와 도구 호출 로그를 메인 채팅 영역에 실시간 렌더링. `on_agent_text` 콜백으로 최근 3블록, 도구 로그 최근 5개 표시. `🔄`/`✅` 마커는 볼드 강조
- 자체 채팅 UI를 가진 앱(TestMancer 등)은 `page.py`에서 동일한 `progress_container` 패턴을 직접 구현

**앱 스위처 UI(`app.py`) 수정:**
- 메인 화면 상단의 카드 바로 앱 전환. 각 앱은 `st.button(type="primary"|"secondary")`로 렌더링
- 카드 스타일은 `core/styles.py`의 `.app-card-btn` CSS 클래스로 정의
- 앱 전환 시 `st.session_state["_selected_app_id"]` 변경 후 `st.rerun()`
- 런처는 `current_page.render_sidebar(prefix)`와 `current_page.render_main(prefix, sidebar_cfg)`만 호출

**앱 page 모듈(`apps/{name}/page.py`) 작성 규칙:**
- 3개 필수 함수: `init_app_session(prefix)`, `render_sidebar(prefix) -> dict`, `render_main(prefix, sidebar_cfg)`
- 모든 Streamlit 위젯 `key`에 `{prefix}_` 접두사 필수 (앱 간 위젯 충돌 방지)
- 세션 상태는 `core/session.py`의 `ns(prefix, key)`, `get_state`, `set_state` 사용 권장
- `core/` 유틸리티 재사용 가능하나 필수 아님. 앱이 완전 자체 구현 가능

# 🚀 DevAgent Platform

**멀티 앱 아키텍처 기반 자율 코딩 에이전트 플랫폼 (Streamlit UI + FastAPI 파일 서버)**

Ollama, Gemini API, vLLM을 지원하는 멀티 앱 플랫폼입니다.
메인 화면 상단의 카드 바에서 앱(DeepAssist, TestMancer 등)을 선택하여 전환할 수 있으며, 각 앱의 대화 상태가 독립적으로 보존됩니다.
사용자의 요청을 분석하여 To-do 계획을 수립하고, 순차적으로 실행한 뒤, 검증까지 자동으로 수행합니다.
멀티유저 환경을 지원하며, 사용자 IP별로 독립 워크스페이스를 자동 할당합니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **멀티 앱 플랫폼** | 메인 화면 카드 바에서 앱(DeepAssist, TestMancer 등) 전환. 앱별 세션 독립 보존 |
| **앱 자동 레지스트리** | `apps/{name}/config.py`에 `APP_CONFIG` 추가만으로 새 앱 자동 등록 |
| **멀티 LLM 통합 구조** | **Claude Agent SDK**를 단일 코어로 통일. 백엔드 전략 패턴으로 Proxy(claude-code-proxy) / Ollama Native / vLLM 3종 지원 |
| **백엔드 모드 전환** | 사이드바에서 프로바이더 선택 (Ollama/Gemini/vLLM). Ollama는 `auto`/`proxy`/`native` 모드 추가 지원 |
| **커스텀 룰 자동화** | 워크스페이스 내 `DeepAssist.md`가 있으면 AI 프롬프트에 자동 적용 |
| **에이전트 도구 시스템** | Claude MCP 서버 및 빌트인 도구(Bash, Read, Write, Edit, Glob 등) 전면 사용 |
| **FAISS + BM25 하이브리드 RAG** | MarkdownRAG + CodeRAG 통합. 쿼리 기반 DB 자동 선택, 문서/프로젝트별 식별, tree-sitter AST 코드 청킹, 서브 청크 재조립, FAISS(60%)+BM25(40%) 앙상블 |
| **멀티유저 워크스페이스** | FastAPI 서버가 IP별 독립 워크스페이스를 자동 생성·관리 (24시간 미활성 시 자동 삭제) |
| **파일 관리 UI** | 워크스페이스 탭에서 파일 업로드·다운로드·생성·편집·삭제 가능 |
| **자동 계획 수립** | 프롬프트 가로채기 방식(시스템 지시어)으로 `Todo List(Plan) 작성 → 실행` 강제 수행 |
| **실시간 UI 로그** | 에이전트의 도구 호출 및 처리 진행 상태를 실시간 출력 |
| **대형 파일 처리** | Claude Agent SDK의 기본 도구를 이용해 안전하게 처리 |
| **2가지 실행 모드** | 에이전트 모드 (계획수립+실행) / 단순 채팅 모드 (프록시 없이 직접 호출, 빠른 응답) |

---

## 설치 & 실행

### 1. Ollama 설치 및 모델 다운로드

```bash
# 대화용 모델 다운로드 (에이전트 모드는 tool calling 지원 모델 권장)
ollama pull qwen3:8b          # 에이전트 모드 권장 (8B+, tool calling 지원)
ollama pull qwen3-coder       # 코딩 특화 모델 (권장)

# RAG용 다국어 임베딩 모델 다운로드
ollama pull bge-m3:latest

# Ollama 서버 실행 (별도 터미널)
ollama serve
```

> **에이전트 모드 모델 요구사항**: tool calling을 지원하는 모델이 필요합니다.
> - 권장: `qwen3:8b`, `qwen3:32b`, `qwen3-coder`, `llama3.1`, `glm-4.7`
> - 비권장: 2B 이하 소형 모델, 비전 전용 모델 (tool calling 불안정)
>
> **Ollama Native 모드** (실험적): Ollama v0.14+ 필요. `native` 백엔드 모드 선택 시 프록시 없이 Anthropic API 호환 모드로 직접 연결됩니다.

### 2. Python 의존성 설치

```bash
pip install -r requirements.txt
```

### 2-1. claude-code-proxy 설치 (Ollama/Gemini/OpenRouter/vLLM 에이전트 모드 필수)

Claude Agent SDK와 로컬/외부 LLM을 연결하는 전용 프록시입니다. Claude 이외 프로바이더로 에이전트 모드를 사용하려면 반드시 설치해야 합니다.

```bash
pip install claude-code-proxy
```

설치 후 `claude-code-proxy` CLI가 PATH에 등록되었는지 확인:

```bash
claude-code-proxy --help
```

> **동작 원리**: Claude Agent SDK → claude-code-proxy(포트 8082) → Ollama/Gemini/OpenRouter/vLLM. 프록시가 Claude API 형식을 각 프로바이더의 API 형식으로 자동 변환합니다. Claude 특수 파라미터(`context_management` 등)도 프록시가 처리합니다.

### 2-2. vLLM 설치 (선택사항 — 프로덕션 고성능 로컬 LLM 서빙)

vLLM은 GPU 환경에서 고성능 LLM 서빙을 제공합니다. Guided Decoding(Outlines)을 통해 tool call JSON의 100% 유효성을 보장합니다.

```bash
# vLLM 설치 (CUDA 필수)
pip install vllm

# vLLM 서버 시작 (Guided Decoding + Tool Calling 활성화)
vllm serve Qwen/Qwen3-8B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --guided-decoding-backend outlines \
  --max-model-len 32768

# 서버 확인
curl http://localhost:8000/v1/models
```

| vLLM 옵션 | 설명 |
|-----------|------|
| `--enable-auto-tool-choice` | 모델이 tool calling을 자동 선택 |
| `--tool-call-parser hermes` | Hermes 형식 tool call 파싱 (qwen3, llama3.1 등 호환) |
| `--guided-decoding-backend outlines` | Outlines 기반 constrained generation으로 JSON 유효성 보장 |
| `--max-model-len 32768` | 최대 컨텍스트 길이 |
| `--quantization awq` | AWQ 양자화 모델 사용 시 (GPU 메모리 절약) |

> **Ollama vs vLLM 비교**:
> - **Ollama**: 간편한 설치, CPU/GPU 자동 감지, 개발/테스트 환경에 적합
> - **vLLM**: GPU 필수, PagedAttention으로 높은 동시 처리량, Guided Decoding으로 tool call 안정성 보장, 프로덕션 환경에 적합

### 3. 환경 변수 설정

```bash
# .env 파일을 생성하고 필요한 환경 변수를 설정하세요
# 아래 '환경 변수 설정' 섹션을 참고하세요
```

### 4. 실행

```bash
# ▶ 원클릭 실행 (FastAPI + Streamlit 동시 기동)
# Mac/Linux:
chmod +x start.sh && ./start.sh

# Windows:
start.bat

# ▶ 수동 실행 (터미널 2개)
python server.py          # 터미널 1: FastAPI 파일 서버 (포트 8000)
streamlit run app.py      # 터미널 2: Streamlit UI (포트 8501)
```

| 서비스 | URL | 설명 |
|--------|-----|------|
| 웹 UI | `http://localhost:8501` | Streamlit 메인 화면 |
| API 문서 | `http://localhost:8000/docs` | FastAPI Swagger UI |

---

## 파일 서버 API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| `GET` | `/api/session` | 현재 세션 ID, 워크스페이스 경로, 사용량 반환 |
| `GET` | `/api/files/listdir?path=` | 지정 경로의 파일 목록 조회 |
| `POST` | `/api/files/upload` | 파일 업로드 (단일 파일 최대 **100MB**) |
| `GET` | `/api/files/download/{filename}` | 파일 다운로드 |
| `DELETE` | `/api/files/{filename}` | 파일 삭제 |
| `GET` | `/api/files/read/{filename}` | 텍스트 파일 내용 읽기 |
| `POST` | `/api/files/write` | 파일 내용 쓰기/생성 |
| `GET` | `/api/health` | 서버 상태 확인 |

> **용량 제한**: 단일 파일 100MB · 워크스페이스 전체 100MB

---

## 워크스페이스 파일 관리 (UI)

「📁 워크스페이스」 탭에서 다음을 수행할 수 있습니다:

| 기능 | 설명 |
|------|------|
| **파일 목록** | 이름·크기·수정일 표시, 클릭 시 인라인 편집기 오픈 |
| **새 파일** | 파일명 입력 후 생성, 즉시 편집 가능 |
| **업로드** | 허용 확장자 파일을 현재 워크스페이스로 업로드 |
| **다운로드** | `⬇` 버튼으로 개별 파일 다운로드 |
| **삭제** | `🗑` 버튼으로 개별 파일 삭제 |
| **인라인 편집기** | 우측 패널에서 직접 편집 후 저장 |

허용 확장자: `md txt py json yaml yml csv html js ts sh env toml ini cfg log`

---

## RAG 사용법

마크다운(.md) 기술문서와 소스코드(.py/.c/.cpp/.java)를 지식 베이스로 구축하여 에이전트가 참조할 수 있습니다.
소스 경로의 파일 확장자를 자동 분석하여 MarkdownRAG 또는 CodeRAG를 선택합니다.

### 검색 동작 방식

| 상황 | 동작 | 예시 |
|------|------|------|
| `db_path` 명시 | 지정된 DB만 검색 | `search_knowledge(query, db_path="/knowledge/nvme23")` |
| `db_path` 미지정 + 쿼리에 DB명 포함 | 관련 DB만 자동 선택 | "XXX92K 코드 만들어줘" → XXX92K DB만 검색 (XXX93K 제외) |
| `db_path` 미지정 + 매칭 없음 | 전체 DB Fan-out 검색 | "telemetry 로그 구조" → 모든 DB에서 검색 후 RRF 병합 |

- `list_knowledge_dbs()`: 구축된 전체 DB 목록 조회 (유형/이름/경로)
- 독립 토큰 매칭으로 서브스트링 오매칭 방지 ("XXX"가 "XXX92k" 안에서 매칭되지 않음)
- 문서별 `doc_name`, 프로젝트별 `project_name`이 검색 결과에 출처로 표시됨

### MarkdownRAG 청킹 전략

| 분할 단계 | 동작 | 비고 |
|-----------|------|------|
| **프론트매터 스킵** | 목차(TOC), 표지 등 헤더만 나열된 구간을 자동 감지하여 건너뜀 | 헤더 뒤 본문 100자 이상인 첫 헤더부터 청킹 시작 |
| **1차: 헤더 경계** (`#`~`######`) | 1000자 이상 누적 후 헤더를 만나면 분할 | 섹션 계층 경로를 메타데이터에 기록 |
| **노이즈 필터링** | `---` (수평선), `<!-- page: N -->` (페이지 마커) 스킵 | 페이지 번호는 메타데이터에만 기록 |
| **2차: 최대 크기 제한** (3000자) | 초과 청크를 테이블 행(`\|`) > 단락(`\n\n`) > 줄바꿈(`\n`) 순으로 재분할 | 구조적 문맥(접두사+마크다운 헤더+테이블 헤더행+구분선) 자동 주입 |

### 용어 인덱스 (Term Index)

기술문서에서 흔한 구조화된 용어를 청킹 시 자동 추출하여 별도 인덱스에 저장합니다. 이를 통해 앙상블 검색(FAISS+BM25)으로 찾기 어려운 약어·ID도 정확하게 검색할 수 있습니다.

| 용어 유형 | 패턴 예시 | 추출 결과 |
|-----------|-----------|-----------|
| **Requirement ID** | `TEL-6`, `SEC-3`, `FWUP-5` | 그대로 키로 등록 |
| **약어 정의** | `Submission Queue (SQ)` | `SQ` + `SUBMISSION QUEUE` 둘 다 등록 |
| **기술 용어** | `Management Operation (MO)` | `MO` + `MANAGEMENT OPERATION` 둘 다 등록 |

**오탐 방지:** `UTF-8`, `AES-256`, `JESD218B-02` 등 일반 기술 규격 번호는 `_REQ_ID_EXCLUDE` 제외 목록으로 Requirement ID 오인식을 방지합니다.

**빈도 기반 분기:** 용어가 10개 이하 청크에만 등장하는 희귀 용어일 때만 직접 검색합니다. `Submission Queue`처럼 수백 회 등장하는 범용 용어는 직접 검색이 역효과이므로 FAISS+BM25 앙상블에 맡깁니다.

**검색 시 동작:**

```
쿼리: "Explain to me about TEL-3"       (희귀 용어)
  ↓ _extract_terms → {"TEL-3"}
  ↓ term_index["TEL-3"] → 매핑 청크 2개 (≤ 10) → 직접 검색
  ├─ 용어 인덱스: TEL-3이 포함된 청크 직접 반환 (정확 매칭)
  ├─ 앙상블 검색: 시맨틱/키워드 보충
  └─ 병합: 용어 매칭 우선 + 앙상블 보충 + 중복 제거

쿼리: "Submission Queue의 동작 원리"    (범용 용어)
  ↓ _extract_terms → {"SUBMISSION QUEUE"}
  ↓ term_index["SUBMISSION QUEUE"] → 매핑 청크 200개 (> 10) → 직접 검색 스킵
  └─ 앙상블 검색만 사용: FAISS(60%) + BM25(40%)
```

### BM25 전처리기

Requirement ID의 하이픈을 보존하여 `TEL-6`과 `TEL-3`을 구분합니다:

| 입력 | 처리 결과 | 설명 |
|------|-----------|------|
| `TEL-6` | `tel-6` | 하이픈 보존, 단일 토큰 유지 |
| `NVMe-oF` | `nvme of` | 일반 하이픈은 공백으로 치환 |

### CodeRAG — 소스코드 전용

| 기능 | 설명 |
|------|------|
| **tree-sitter AST 청킹** | 함수/클래스 경계를 정확히 보존 (미설치 시 regex 폴백) |
| **Python 데코레이터 인식** | `@test_case("TEL-2")` 등 데코레이터가 함수와 함께 청킹 |
| **3단계 청크 체계** | L1 파일 요약, L2 함수/클래스, L3 서브 청크 |
| **서브 청크 재조립** | 대형 함수 분할 후 검색 시 `function_id`로 완전한 함수 복원 |
| **코드 내 Req ID 추출** | `TEL-2` (하이픈) + `TEL_2` (언더스코어) 모두 인식 |
| **5종 인덱스** | symbol, req_id, file_path, function_id, file_manifest |
| **언어 확장** | `_LANGUAGE_CONFIG`에 한 줄 추가로 새 언어 지원 |

지원 언어: Python, C, C++, Java (tree-sitter 그래머 설치 시)

### 기본 사용

```python
# ── MarkdownRAG (doc_name으로 문서 식별) ──
from rag import MarkdownRAG

rag = MarkdownRAG(db_store_path="/path/to/knowledge/db", doc_name="NVMe 2.3 Base")
rag.build_or_load("/path/to/document.md")  # DB 있으면 로드, 없으면 구축
results = rag.retrieve("TEL-6", top_k=4)   # Requirement ID 직접 검색

# ── CodeRAG (project_name으로 프로젝트 식별) ──
from rag import CodeRAG

code_rag = CodeRAG(db_store_path="/path/to/code/db", project_name="XXX92K")
code_rag.build_or_load("/path/to/project/src")  # 소스코드 폴더 인덱싱
results = code_rag.retrieve("TEL-2 관련 평가", top_k=4)    # Req ID + 도메인 검색
results = code_rag.retrieve("main.cpp 파일은 어떤 기능?")   # 파일 요약 검색
results = code_rag.retrieve("allocate_block", top_k=4)      # 심볼 검색
```

### DB 저장 구조

```
# MarkdownRAG DB
/knowledge/{db_name}/
├── faiss_index/          # FAISS 벡터 DB
├── bm25_retriever.pkl    # BM25 인덱스
├── term_index.pkl        # 용어 인덱스 (Req ID + 약어 → Document 매핑)
└── doc_meta.pkl          # 문서 식별 (doc_name)

# CodeRAG DB
/knowledge/{db_name}/
├── faiss_index/              # FAISS 벡터 DB
├── bm25_retriever.pkl        # BM25 인덱스
├── symbol_index.pkl          # 심볼명 → 정의 청크 매핑
├── req_id_index.pkl          # Req ID → 포함 청크 매핑
├── file_path_index.pkl       # 파일 경로 → L1 요약 청크 매핑
├── function_id_index.pkl     # function_id → 서브 청크 리스트 (재조립용)
├── file_manifest.pkl         # 파일 경로 → mtime (증분 빌드용)
└── project_meta.pkl          # 프로젝트 식별 (project_name, project_root)
```

### 테스트 확장

`rag.py`의 `__main__` 섹션에서 `TESTS` 리스트에 항목을 추가하면 됩니다:

```python
TESTS = [
    # ── 마크다운 RAG ──
    {
        "type": "markdown",
        "label": "문서 이름",
        "source": "/path/to/document.md",
        "db_path": "/path/to/knowledge/db",
        "queries": [{"tag": "테스트명", "query": "검색 쿼리"}],
    },
    # ── 코드 RAG ──
    {
        "type": "code",
        "label": "프로젝트 이름",
        "source": "/path/to/project/src",
        "db_path": "/path/to/knowledge/code_db",
        "queries": [
            {"tag": "Req ID 검색",  "query": "TEL-2 관련 평가"},
            {"tag": "파일 요약",    "query": "main.cpp 파일은 어떤 기능?"},
            {"tag": "심볼 검색",    "query": "allocate_block"},
        ],
    },
]
```

### 현재 등록된 문서

| 유형 | 문서 | DB 경로 | 테스트 쿼리 |
|------|------|---------|-------------|
| markdown | NVMe Base Spec 2.3 | `/Users/howard/Project/knowledge/nvme23` | 4개 (큐 메커니즘, RESERVS, MO, SQ) |
| markdown | OCP Datacenter NVMe SSD 2.6 | `/Users/howard/Project/knowledge/ocp26` | 13개 (DWPD, 텔레메트리, TEL-3/6, SEC-5, FWUP-5, 자연어+ID 등) |

### 독립 실행

```bash
python3 rag.py              # 전체 테스트 (markdown + code)
python3 rag.py markdown     # 마크다운 RAG만 테스트
python3 rag.py code         # 코드 RAG만 테스트
```

---

## 환경 변수 설정 (.env)

| 변수 | 설명 |
|------|------|
| `EMBEDDING_PROVIDER` | 임베딩 프로바이더 (`ollama` 또는 `gemini`) |
| `OLLAMA_BASE_URL` | Ollama 서버 주소 (기본: `http://localhost:11434`) |
| `OLLAMA_EMBEDDING_MODEL` | Ollama 임베딩 모델명 (예: `bge-m3:latest`) |
| `GEMINI_API_KEY` | Gemini API Key |
| `GEMINI_EMBEDDING_MODEL` | Gemini 임베딩 모델명 (예: `models/text-embedding-004`) |
### 백엔드 연결 구조

| 프로바이더 | 백엔드 전략 | 프록시 | 연결 흐름 |
|-----------|-----------|--------|----------|
| **Ollama** (proxy) | Proxy | claude-code-proxy | SDK → proxy → Ollama |
| **Ollama** (native) | OllamaNative | 없음 | SDK → Ollama `/v1/messages` 직접 |
| **Ollama** (auto) | Native → Proxy 폴백 | 조건부 | Native 시도 → 실패 시 Proxy |
| **Gemini API** | Proxy | claude-code-proxy | SDK → proxy → Gemini |
| **vLLM** | Vllm | claude-code-proxy | SDK → proxy → vLLM OpenAI API |

> **proxy/auto/vLLM 모드**: `pip install claude-code-proxy` 설치 필요
>
> **native 모드**: Ollama v0.14 이상 필요 (`ollama --version`으로 확인). `count_tokens` 엔드포인트 관련 이슈가 있을 수 있음 — 불안정 시 `proxy` 모드 사용 권장
>
> **vLLM**: vLLM 서버가 별도로 실행 중이어야 합니다. GPU + CUDA 환경 필요

---

## 아키텍처

```
dev_agent/
├── app.py                      # 멀티 앱 런처 (앱 카드 바 + 라우팅)
├── core/                       # 공유 유틸리티 (앱이 선택적으로 import)
│   ├── session.py              # 네임스페이스 세션 상태 (앱별 독립)
│   ├── styles.py               # 공통 CSS (앱 카드, 채팅, 사이드바)
│   ├── sidebar.py              # LLM 프로바이더 선택 사이드바
│   ├── chat_ui.py              # 채팅 탭 (콜백, 에이전트 생성, 메시지)
│   └── workspace_ui.py         # 워크스페이스 파일 관리 탭
├── apps/                       # 앱 레지스트리 (자동 검색)
│   ├── __init__.py             # discover_apps() — config + page 자동 검색
│   ├── deep_assist/
│   │   ├── config.py           # DeepAssist APP_CONFIG
│   │   └── page.py             # 사이드바/메인 렌더링 (core/ 재사용)
│   └── test_mancer/
│       ├── config.py           # TestMancer APP_CONFIG
│       └── page.py             # 자체 사이드바/메인 (테스트 특화 UI)
├── models.py                   # 핵심 데이터 모델 (Task, Plan, ToolCallRecord)
├── llm_clients.py              # LLM 클라이언트 (Ollama, Gemini)
├── backend_strategy.py         # 백엔드 전략 (Direct/Proxy/Native/vLLM 4종)
├── agent.py                    # Claude Agent SDK Runner (전략 위임)
├── mcp_server.py               # MCP 서버 (RAG 도구 + 웹 검색)
├── rag.py                      # FAISS + BM25 하이브리드 RAG (MarkdownRAG + CodeRAG)
├── server.py                   # FastAPI 파일/워크스페이스 관리 서버
├── .env                        # 환경변수 (API Key, 임베딩 모델 등)
├── start.sh                    # Mac/Linux 실행 스크립트
├── nginx.conf                  # Nginx 리버스 프록시 설정
├── requirements.txt
├── CLAUDE.md                   # 프로젝트 가이드 (Claude Code용)
└── workspaces/                 # IP별 자동 생성 워크스페이스 (24시간 후 자동 삭제)
    ├── {session_A}/
    └── {session_B}/
```

### 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 사용자 브라우저                                                               │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ Streamlit (app.py → apps/{name}/page.py) :8501                                │
│  ├─ 앱 스위처 카드 바: DeepAssist / TestMancer / ... (앱별 세션 독립)            │
│  ├─ 앱별 독립 사이드바: page.render_sidebar(prefix)                              │
│  ├─ 앱별 독립 메인 화면: page.render_main(prefix, sidebar_cfg)                   │
│  └─ core/: 공유 유틸리티 (앱이 선택적으로 import)                                │
└──────────┬───────────────────────────────┬──────────────────────────────────┘
           │ 에이전트 실행                    │ REST API (파일 관리)
           ↓                                ↓
┌──────────────────────────────────┐  ┌────────────────────────────────────────┐
│ ClaudeAgentRunner (agent.py)     │  │ FastAPI (server.py) :8000              │
│  ├─ Claude Agent SDK 구동         │  │  ├─ IP 기반 세션/워크스페이스 자동 할당   │
│  ├─ BackendStrategy 선택/위임     │  │  ├─ 파일 CRUD (단일 100MB / 전체 100MB) │
│  │  ├─ Proxy: claude-code-proxy   │  │  ├─ 비활성 워크스페이스 30분 주기 삭제   │
│  │  ├─ Native: Ollama 직접 연결   │  │  └─ Path Traversal 공격 방지           │
│  │  └─ vLLM: vLLM+proxy 경유     │  └────────────────────────────────────────┘
│  ├─ 시스템 프롬프트 주입           │
│  │  └─ DeepAssist.md 자동 로드    │
│  └─ Todo List 자동 수립/추적      │
└──────────┬───────────────────────┘
           │ MCP (stdio)
           ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ MCP Server (mcp_server.py)                                                   │
│                                                                              │
│  ┌─ list_knowledge_dbs ─┐  ┌─ search_knowledge ──┐  ┌─ search_web_and_  ─┐  │
│  │ 구축된 DB 목록 조회    │  │ 하이브리드 RAG 검색  │  │ scrape             │  │
│  └──────────────────────┘  │                      │  │ 웹 검색+스크래핑   │  │
│                            │  쿼리 기반 DB 자동선택 │  └──────────────────┘  │
│  ┌─ build_knowledge_db ─┐  │  ┌─────────────────┐ │                        │
│  │ DB 구축 (소스유형 자동│  │  │_match_dbs_by    │ │  ┌─ 빌트인 도구 ─────┐  │
│  │ 감지: md→Markdown     │  │  │_query           │ │  │ Bash, Read, Write │  │
│  │       code→CodeRAG)  │  │  │ 독립 토큰 매칭   │ │  │ Edit, Glob, Grep  │  │
│  └──────────────────────┘  │  └─────────────────┘ │  └──────────────────┘  │
│                            └──────────┬───────────┘                        │
└───────────────────────────────────────┼────────────────────────────────────┘
                                        ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ RAG Engine (rag.py)                                                          │
│                                                                              │
│  ┌─ BaseRAG ──────────────────────────────────────────────────────────────┐  │
│  │  _init_embeddings (Ollama / Gemini)                                    │  │
│  │  _save_vector_stores / _load_vector_stores (FAISS + BM25)              │  │
│  │  _setup_ensemble (FAISS 60% + BM25 40% → EnsembleRetriever)           │  │
│  └────────────────────────┬─────────────────────────┬─────────────────────┘  │
│                           │                         │                        │
│  ┌─ MarkdownRAG ─────────┐│  ┌─ CodeRAG ───────────┐│                        │
│  │  doc_name 식별         ││  │  project_name 식별   ││                        │
│  │  헤더 경계 청킹         ││  │  tree-sitter AST    ││                        │
│  │  프론트매터 스킵        ││  │  3단계 청크 (L1/2/3) ││                        │
│  │  테이블 헤더 보존       ││  │  데코레이터 인식     ││                        │
│  │  용어 인덱스            ││  │  서브 청크 재조립    ││                        │
│  │  (Req ID + 약어)       ││  │  5종 인덱스          ││                        │
│  │  doc_meta.pkl          ││  │  project_meta.pkl    ││                        │
│  └────────────────────────┘│  └──────────────────────┘│                        │
│                            │                          │                        │
└────────────────────────────┼──────────────────────────┼────────────────────────┘
                             ↓                          ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ Knowledge DB Store (~/.deepassist/knowledge/)                                │
│                                                                              │
│  ┌─ nvme23_xxx/ ──────┐  ┌─ ocp26_xxx/ ──────┐  ┌─ XXX_xxx/ ──────────┐  │
│  │ [MARKDOWN]          │  │ [MARKDOWN]         │  │ [CODE]                 │  │
│  │ doc: NVMe 2.3 Base  │  │ doc: OCP 2.6 SSD   │  │ project: XXX        │  │
│  │ faiss_index/        │  │ faiss_index/        │  │ faiss_index/           │  │
│  │ bm25_retriever.pkl  │  │ bm25_retriever.pkl  │  │ bm25_retriever.pkl     │  │
│  │ term_index.pkl      │  │ term_index.pkl      │  │ symbol_index.pkl       │  │
│  │ doc_meta.pkl        │  │ doc_meta.pkl        │  │ req_id_index.pkl       │  │
│  └─────────────────────┘  └─────────────────────┘  │ file_path_index.pkl    │  │
│                                                     │ function_id_index.pkl  │  │
│  ┌─ XXXXX_xxx/ ──────┐  ┌─ XXXX_xxx/ ─────┐    │ file_manifest.pkl      │  │
│  │ [CODE]              │  │ [CODE]             │    │ project_meta.pkl       │  │
│  │ project: XXXXX     │  │ project: XXXX    │    └────────────────────────┘  │
│  │ ...                 │  │ ...                │                                │
│  └─────────────────────┘  └─────────────────────┘                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 실행 흐름

```
사용자 브라우저
    ↓
┌──────────────────────────────────────────────────┐
│ app.py (런처) :8501                                │
│  ├─ apps/ 레지스트리에서 앱 자동 검색               │
│  ├─ 상단 카드 바: 앱 선택 (DeepAssist/TestMancer)  │
│  ├─ page.init_app_session(prefix) → 앱별 세션     │
│  ├─ page.render_sidebar(prefix) → 앱별 사이드바   │
│  └─ page.render_main(prefix, cfg) → 앱별 메인     │
└──────────────┬───────────────────────────────────┘
               │ REST API (파일 관리)
┌──────────────▼─────────────────┐
│ FastAPI (server.py) :8000      │
│  ├─ IP 기반 세션/워크스페이스   │
│  ├─ 파일 CRUD (100MB 제한)     │
│  ├─ 비활성 워크스페이스 자동삭제│
│  └─ Path Traversal 보안        │
└────────────────────────────────┘
```

### 새 앱 추가 방법

```bash
mkdir apps/my_app
```

```python
# apps/my_app/__init__.py
"""MyApp 설명"""

# apps/my_app/config.py
APP_CONFIG = {
    "id": "my_app",
    "name": "MyApp",
    "icon": "🚀",
    "description": "새로운 에이전트",
    "tabs": ["chat", "workspace"],
    "default_mode": "agent",
    "custom_css": "",
}

# apps/my_app/page.py — 3개 필수 함수
def init_app_session(prefix: str):
    """앱 전용 세션 상태 초기화"""
    from core.session import init_session
    init_session(prefix)  # 기본 세션 + 앱 전용 키 추가 가능

def render_sidebar(prefix: str) -> dict:
    """앱 전용 사이드바 → 설정 dict 반환"""
    ...

def render_main(prefix: str, sidebar_cfg: dict):
    """앱 전용 메인 화면 렌더링"""
    ...
```

- `core/` 유틸리티 재사용 가능 (선택사항). 완전 자체 구현도 가능
- 재시작 시 `discover_apps()`가 `config.py` + `page.py`를 자동 검색하여 카드 바에 등록

### MarkdownRAG 데이터 흐름

```
.md 파일 입력
    ↓
[_find_content_start] → 프론트매터(목차/표지) 자동 스킵
    ↓
[_split_md_by_header_boundary] → 헤더 경계 기준 1차 청킹
    ├─ 헤더(#~######) 경계에서 분할
    ├─ 테이블 헤더행+구분선 보존
    └─ --- / <!-- page: N --> 스킵
    ↓
[RecursiveCharacterTextSplitter] → 3000자 초과 2차 분할
    └─ [_extract_chunk_context] → 구조적 문맥 자동 주입
    ↓
[FAISS + BM25 + Term Index] → 벡터 저장 + 용어 인덱스
    ↓
검색: Term Index 직접 매칭 + Ensemble 보충 → 결과 반환
```

### CodeRAG 데이터 흐름

```
소스코드 폴더 입력
    ↓
[tree-sitter AST 파싱] → 언어별 그래머 자동 선택 (미설치 시 regex 폴백)
    ↓
[L1 파일 요약 청크] → 파일 경로, 라인 수, import, Req ID, 전체 선언 목록
    ↓
[L2 함수/클래스 청크] → AST 노드 단위, 계층 경로 + 시그니처 접두사
    ├─ 소형 선언 (200자 미만) → 인접 항목 병합 (declarations 청크)
    ├─ Python 데코레이터 → decorated_definition 노드로 함수와 함께 포함
    └─ 대형 함수 (4000자 초과) → L3 서브 청크 분할 (function_id로 연결)
    ↓
[코드 내 Req ID 추출] → TEL-2 (하이픈) + TEL_2 (언더스코어) 모두 정규화
    ↓
[FAISS + BM25 + 5종 인덱스] → 벡터 저장 + symbol/req_id/file_path/function_id/manifest
    ↓
검색 쿼리 입력
    ↓
[_extract_code_terms] → 파일 경로 / Req ID / 심볼 추출
    ↓
Tier 1: 직접 검색 (file_path_index, req_id_index, symbol_index)
Tier 2: Ensemble 검색 (FAISS 60% + BM25 40%)
Tier 3: 서브 청크 재조립 (function_id_index → 형제 청크 합침)
    ↓
완전한 함수 단위 결과 반환
```

### MCP 검색 흐름 (search_knowledge)

```
사용자 쿼리 (db_path 미지정)
    ↓
[_list_all_dbs] → ~/.deepassist/knowledge/ 스캔
    ├── nvme23/     → doc_meta.pkl    → {doc_name: "NVMe 2.3 Base"}
    ├── ocp26/      → doc_meta.pkl    → {doc_name: "OCP 2.6 SSD"}
    ├── XXX92K/     → project_meta.pkl → {project_name: "XXX92K"}
    └── XXX93K/     → project_meta.pkl → {project_name: "XXX93K"}
    ↓
[_match_dbs_by_query] → 쿼리에서 DB명 키워드 독립 토큰 매칭
    ├─ "XXX92K 코드 만들어줘" → "XXX92k" 매칭 → XXX92K DB만 선택
    ├─ "nvme에서 telemetry"  → "nvme" 매칭 → NVMe, OCP DB 선택
    └─ "telemetry 로그 구조" → 매칭 없음 → 전체 4개 DB 검색
    ↓
선택된 DB별 _load_and_search (유형 자동 감지)
    ↓
_rrf_merge → 통합 순위 → 결과 반환 (doc_name/project_name 출처 표시)
```

---

## 서버 배포 (Nginx 리버스 프록시)

`nginx.conf.example` 파일을 참고하여 Nginx 뒤에서 운영할 수 있습니다:

```
사용자 → Nginx (80/443)
           ├─ /api/*  → FastAPI :8000
           └─ /*      → Streamlit :8501
```

---

## 주의사항

- `bash` 도구는 시스템 명령을 직접 실행하므로 **신뢰할 수 있는 환경에서만** 사용하세요
- 작업 디렉토리(`DeepAssist.md`가 있는 폴더)를 올바르게 설정하면 프로젝트 전용 규칙이 자동 적용됩니다
- 임베딩 모델을 변경하면 기존 FAISS DB와 호환되지 않으므로 `knowledge_base/` 폴더를 삭제 후 재구축이 필요합니다
- 파일 서버를 실행하지 않아도 에이전트 기능은 정상 동작합니다 (워크스페이스 탭만 비활성화됨)
- 워크스페이스는 24시간 미활성 시 자동 삭제됩니다. 중요한 파일은 다운로드하여 보관하세요
- **에이전트 모드 + Ollama**: tool calling 지원 모델 필요 (qwen3:8b 이상 권장). 2B 이하 소형 모델은 tool call을 JSON 텍스트로 출력하는 문제 발생 가능
- **Ollama Native 모드**: 실험적 기능. `count_tokens` 엔드포인트 이슈로 서버가 불안정해질 수 있음. 문제 발생 시 `proxy` 모드로 전환
- **vLLM**: GPU + CUDA 환경 필수. `--enable-auto-tool-choice`와 `--guided-decoding-backend outlines` 옵션으로 서버를 시작해야 tool calling이 정상 동작
- **claude-code-proxy**: 에이전트 모드를 사용하려면 반드시 설치 필요 (`pip install claude-code-proxy`)

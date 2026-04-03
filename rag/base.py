"""
rag/base.py — BaseRAG 공통 인프라

임베딩 모델 초기화, FAISS/BM25 벡터 스토어 관리, 앙상블 검색 설정 등
MarkdownRAG와 CodeRAG의 공통 기반 클래스.
"""

import logging
import os
import pickle
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:
    GoogleGenerativeAIEmbeddings = None

from config import (
    EMBEDDING_PROVIDER, OLLAMA_DEFAULT_URL, OLLAMA_EMBEDDING_MODEL,
    GEMINI_EMBEDDING_MODEL, GEMINI_API_KEY,
)

logger = logging.getLogger(__name__)


class BaseRAG:
    """MarkdownRAG와 CodeRAG의 공통 기반 클래스.

    임베딩 모델 초기화, FAISS/BM25 벡터 스토어 관리, 앙상블 검색 설정 등
    양쪽 RAG에서 동일하게 사용하는 인프라를 제공한다.
    """

    def __init__(self, db_store_path: str = "./knowledge_base",
                 embedding_model_override: Optional[str] = None):
        """BaseRAG 초기화.

        Args:
            db_store_path: DB를 저장/로드할 디렉토리 경로
            embedding_model_override: 기본 임베딩 모델 대신 사용할 모델명
        """
        self.db_store_path = db_store_path
        self.faiss_path = os.path.join(self.db_store_path, "faiss_index")
        self.bm25_path = os.path.join(self.db_store_path, "bm25_retriever.pkl")

        self.vector_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None

        self.embeddings = self._init_embeddings(embedding_model_override)

    def _init_embeddings(self, model_override: Optional[str] = None):
        """임베딩 모델을 초기화한다.

        config.py의 EMBEDDING_PROVIDER에 따라 Ollama 또는 Gemini 임베딩을 사용.
        """
        provider = EMBEDDING_PROVIDER.lower()
        if provider == "gemini":
            if GoogleGenerativeAIEmbeddings is None:
                raise ImportError("langchain-google-genai 패키지가 설치되지 않았습니다.")
            api_key = GEMINI_API_KEY
            if not api_key:
                raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")
            model = model_override or GEMINI_EMBEDDING_MODEL
            if not model:
                raise ValueError("GEMINI_EMBEDDING_MODEL 값이 설정되지 않았습니다.")
            return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        else:
            base_url = OLLAMA_DEFAULT_URL
            model = model_override or OLLAMA_EMBEDDING_MODEL
            if not model:
                raise ValueError("OLLAMA_EMBEDDING_MODEL 값이 설정되지 않았습니다.")
            return OllamaEmbeddings(base_url=base_url, model=model)

    def _setup_ensemble(self, faiss_k: int = 5, bm25_k: int = 5,
                        weights: Optional[List[float]] = None):
        """FAISS와 BM25를 묶어 Hybrid Retriever를 구성한다."""
        if weights is None:
            weights = [0.4, 0.6]
        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": faiss_k})
        self.bm25_retriever.k = bm25_k
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=weights,
        )

    # FAISS 배치 빌드 크기 (대규모 문서셋에서 메모리 효율 향상)
    FAISS_BATCH_SIZE = 10000

    def _save_vector_stores(self, all_splits: List[Document],
                            bm25_preprocess_func=None):
        """FAISS 벡터 DB와 BM25 인덱스를 구축하고 디스크에 저장한다."""
        logger.info(f"총 {len(all_splits)}개의 청크(Chunk)로 분할되었습니다. 임베딩 진행 중...")
        os.makedirs(self.db_store_path, exist_ok=True)

        # 배치 단위로 FAISS에 추가 (메모리 효율)
        vs = None
        for i in range(0, len(all_splits), self.FAISS_BATCH_SIZE):
            batch = all_splits[i:i + self.FAISS_BATCH_SIZE]
            if vs is None:
                vs = FAISS.from_documents(batch, self.embeddings)
            else:
                vs.add_documents(batch)
            if len(all_splits) > self.FAISS_BATCH_SIZE:
                logger.info(f"  FAISS 배치 {i // self.FAISS_BATCH_SIZE + 1}/"
                             f"{(len(all_splits) - 1) // self.FAISS_BATCH_SIZE + 1} 완료")
        self.vector_store = vs
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

"""LanceDB Vector Engine for Second Brain Memory.

Provides two-stage semantic search over historical conversation turns.
Stage 1: Dense vector recall using BAAI/bge-small-en-v1.5 (CUDA-accelerated).
Stage 2: Precision Cross-Encoder reranking (ms-marco-MiniLM-L-6-v2) for deep semantic relevance.
Implements a strict Privacy Guardrail via the `is_local_only` flag.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from core.logging_setup import get_logger
from core.db import _ensure_dir

_log = get_logger("memory_vector")

EMBEDDING_DIM = 384
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MEMORY_SCHEMA = pa.schema([
    pa.field("turn_id", pa.int32()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("is_local_only", pa.bool_()),
    pa.field("timestamp", pa.string())
])

def _get_device() -> str:
    """Detect CUDA GPU availability, fallback to CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"

class VectorEngine:
    """Local GPU-accelerated two-stage vector store for semantic memory retrieval."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        _ensure_dir(self.db_path)
        self._db = None
        self._table = None
        self._model = None
        self._reranker = None
        self._warmup_task = None
        self._model_lock = asyncio.Lock()
        self._reranker_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._device = _get_device()

    async def initialize(self) -> None:
        """Open LanceDB and ensure the memories table exists."""
        try:
            self._db = await asyncio.to_thread(lancedb.connect, str(self.db_path))
            
            # Create or open table
            try:
                self._table = await asyncio.to_thread(self._db.open_table, "memories")
            except Exception:
                self._table = await asyncio.to_thread(
                    self._db.create_table, "memories", schema=MEMORY_SCHEMA
                )
                
            # Warm up models in background task so first user turn has zero cold-start delay
            self._warmup_task = asyncio.create_task(self.warmup())
            _log.info("vector_engine_initialized", path=str(self.db_path), device=self._device)
        except Exception as e:
            _log.error("vector_engine_init_failed", error=str(e))
            raise

    async def close(self) -> None:
        """Cancel any in-flight background warmup task."""
        if self._warmup_task is not None and not self._warmup_task.done():
            self._warmup_task.cancel()
            try:
                await self._warmup_task
            except (asyncio.CancelledError, Exception):
                pass

    async def warmup(self) -> None:
        """Pre-load embedding and reranker models into GPU memory."""
        try:
            await self._get_embedding_model()
            await self._get_reranker_model()
        except Exception as e:
            _log.warning("vector_engine_warmup_failed", error=str(e))

    async def _get_embedding_model(self) -> SentenceTransformer:
        """Lazy-load the BGE sentence transformer model into memory/GPU."""
        if self._model is None:
            async with self._model_lock:
                if self._model is None:
                    _log.debug("loading_sentence_transformer", model=EMBEDDING_MODEL_NAME, device=self._device)
                    def _load_model():
                        m = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self._device)
                        if self._device == "cuda":
                            m.half()
                        return m
                    self._model = await asyncio.to_thread(_load_model)
        return self._model

    async def _get_reranker_model(self) -> CrossEncoder:
        """Lazy-load the Cross-Encoder reranker model into memory/GPU."""
        if self._reranker is None:
            async with self._reranker_lock:
                if self._reranker is None:
                    _log.debug("loading_cross_encoder_reranker", model=RERANKER_MODEL_NAME, device=self._device)
                    def _load_reranker():
                        r = CrossEncoder(RERANKER_MODEL_NAME, device=self._device)
                        if self._device == "cuda":
                            r.model.half()
                        return r
                    self._reranker = await asyncio.to_thread(_load_reranker)
        return self._reranker

    async def embed(self, text: str) -> list[float]:
        """Generate a 384-dim normalized vector using BGE on CUDA/CPU."""
        stripped = text.strip()
        if not stripped:
            return [0.0] * EMBEDDING_DIM
            
        model = await self._get_embedding_model()
        def _encode():
            with torch.inference_mode():
                return model.encode(stripped, normalize_embeddings=True)
        vector = await asyncio.to_thread(_encode)
        return vector.tolist()

    async def add_memory(
        self, turn_id: int, text: str, is_local_only: bool, timestamp: datetime | None = None
    ) -> None:
        """Embed a memory and store it in LanceDB."""
        if self._table is None or not text or not text.strip():
            return
            
        timestamp = timestamp or datetime.now()
        vector = await self.embed(text)
        chunk = {
            "turn_id": turn_id,
            "text": text.strip(),
            "vector": vector,
            "is_local_only": is_local_only,
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
        }
        async with self._write_lock:
            await asyncio.to_thread(self._table.add, [chunk])

    async def search(
        self,
        query: str,
        limit: int = 5,
        include_local_only: bool = True,
        max_distance: float = 1.0,
        min_rerank_score: float = -5.0
    ) -> list[dict[str, Any]]:
        """Perform two-stage semantic search: Dense Recall + Cross-Encoder Reranking.
        
        Args:
            query: The search query string.
            limit: Maximum number of final reranked results to return.
            include_local_only: If False, filters out private/local memories.
            max_distance: Maximum L2 distance cutoff for stage-1 candidate recall.
            min_rerank_score: Minimum Cross-Encoder logit score to eliminate irrelevant candidates.
        """
        if self._table is None or not query or not query.strip():
            return []
            
        try:
            query_vector = await self.embed(query)
            
            # Stage 1: Candidate Recall (fetch 3x candidate pool)
            fetch_pool = max(limit * 3, 10)
            search_builder = self._table.search(query_vector).limit(fetch_pool)
            
            if not include_local_only:
                search_builder = search_builder.where("is_local_only = false", prefilter=True)
                
            raw_results = await asyncio.to_thread(search_builder.to_list)
            
            # Filter by stage-1 distance cutoff
            candidates = [r for r in raw_results if r.get("_distance", 2.0) <= max_distance]
            if not candidates:
                return []
                
            # Stage 2: Cross-Encoder Precision Reranking
            try:
                reranker = await self._get_reranker_model()
                pairs = [(query, c["text"]) for c in candidates]
                def _predict():
                    with torch.inference_mode():
                        return reranker.predict(pairs)
                scores = await asyncio.to_thread(_predict)
                
                for candidate, score in zip(candidates, scores):
                    candidate["_rerank_score"] = float(score)
                    
                # Sort by rerank score descending
                candidates.sort(key=lambda x: x.get("_rerank_score", -999.0), reverse=True)
                
                # Filter out candidates with very low cross-encoder confidence
                reranked = [c for c in candidates if c.get("_rerank_score", -999.0) >= min_rerank_score]
                return reranked[:limit]
            except Exception as re_err:
                _log.warning("reranker_fallback_to_distance", error=str(re_err))
                # Fallback: sort by distance if reranker encountered an issue
                candidates.sort(key=lambda x: x.get("_distance", 999.0))
                return candidates[:limit]

        except Exception as e:
            _log.error("vector_engine_search_failed", query=query[:50], error=str(e))
            return []

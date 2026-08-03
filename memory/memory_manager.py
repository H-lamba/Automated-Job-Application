"""
memory/memory_manager.py — Long-term memory for the Career Agent.

Two-layer architecture:
1. SQLite (via SQLAlchemy) — structured records (applied jobs, sessions)
2. ChromaDB (local) — semantic vector search over job descriptions

This lets the agent answer questions like:
- "Have I applied to this job before?" → SQLite URL lookup
- "What companies have I applied to?" → SQLite query
- "Find jobs similar to this one I liked" → ChromaDB similarity
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.logger import logger


class MemoryManager:
    """
    Manages long-term agent memory.

    - Structured lookups (applied URLs, session history) via SQLite
    - Semantic similarity via ChromaDB
    """

    def __init__(
        self,
        db: AsyncSession,
        chroma_path: str = "./data/chroma",
    ) -> None:
        self.db = db
        self._chroma_path = chroma_path
        self._chroma_client: Any = None
        self._job_collection: Any = None

    # ──────────────────────────────────────────────────────────────────────────
    # ChromaDB initialisation (lazy)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_chroma_client(self) -> Any:
        if self._chroma_client is None:
            Path(self._chroma_path).mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=self._chroma_path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.debug("ChromaDB initialised at {}", self._chroma_path)
        return self._chroma_client

    def _get_job_collection(self):
        if self._job_collection is None:
            client = self._get_chroma_client()
            self._job_collection = client.get_or_create_collection(
                name="job_descriptions",
                metadata={"hnsw:space": "cosine"},
            )
        return self._job_collection

    # ──────────────────────────────────────────────────────────────────────────
    # Application history (SQLite)
    # ──────────────────────────────────────────────────────────────────────────

    async def has_applied(self, application_url: str) -> bool:
        """
        Return True if we've previously submitted an application for this URL.

        Checks both the jobs.status and applications.status tables.
        """
        result = await self.db.execute(
            text("""
                SELECT COUNT(*) FROM jobs j
                LEFT JOIN applications a ON a.job_id = j.id
                WHERE j.application_url = :url
                AND (j.status = 'applied' OR a.status = 'submitted')
            """),
            {"url": application_url},
        )
        count = result.scalar() or 0
        return count > 0

    async def get_applied_companies(self) -> list[str]:
        """Return a list of all companies we have applied to."""
        result = await self.db.execute(
            text("SELECT DISTINCT company FROM jobs WHERE status = 'applied'")
        )
        return [row[0] for row in result.fetchall()]

    async def get_application_stats(self) -> dict[str, Any]:
        """Return aggregate statistics about all applications."""
        result = await self.db.execute(
            text("""
                SELECT
                    status,
                    COUNT(*) as count
                FROM jobs
                GROUP BY status
            """)
        )
        stats = {row[0]: row[1] for row in result.fetchall()}
        return {
            "total_discovered": sum(stats.values()),
            "applied": stats.get("applied", 0),
            "queued": stats.get("queued", 0),
            "failed": stats.get("failed", 0),
            "skipped": stats.get("skipped", 0),
            "by_status": stats,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Semantic memory (ChromaDB)
    # ──────────────────────────────────────────────────────────────────────────

    async def embed_job(
        self,
        job_id: str,
        title: str,
        company: str,
        description: str,
        metadata: dict | None = None,
    ) -> None:
        """
        Store a job description in ChromaDB for future similarity search.

        ChromaDB uses its default embedding model (all-MiniLM-L6-v2).
        """
        try:
            collection = self._get_job_collection()
            document = f"{title} at {company}\n\n{description[:2000]}"
            meta = metadata or {}
            meta.update({"job_id": job_id, "company": company, "title": title})

            collection.upsert(
                ids=[job_id],
                documents=[document],
                metadatas=[meta],
            )
            logger.debug("Embedded job '{}' @ {}", title, company)
        except Exception as e:
            logger.warning("Failed to embed job '{}': {}", title, e)

    def find_similar_jobs(
        self,
        query: str,
        n_results: int = 10,
    ) -> list[dict]:
        """
        Find jobs semantically similar to the query string.

        Returns a list of dicts with job metadata and similarity distance.
        """
        try:
            collection = self._get_job_collection()
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["metadatas", "distances"],
            )
            jobs = []
            for meta, distance in zip(
                results["metadatas"][0],
                results["distances"][0],
                strict=True,
            ):
                jobs.append({**meta, "similarity": 1.0 - distance})
            return jobs
        except Exception as e:
            logger.warning("Similarity search failed: {}", e)
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # Key-value store (simple session memory)
    # ──────────────────────────────────────────────────────────────────────────

    def remember(self, key: str, value: Any) -> None:
        """Store a short-term key-value pair in the ChromaDB metadata store."""
        # For simple key-value, we store in a separate collection
        try:
            client = self._get_chroma_client()
            kv_collection = client.get_or_create_collection("agent_memory")
            kv_collection.upsert(
                ids=[key],
                documents=[str(value)],
                metadatas=[{"key": key}],
            )
        except Exception as e:
            logger.warning("Memory store failed for key '{}': {}", key, e)

    def recall(self, key: str) -> str | None:
        """Retrieve a stored key-value pair."""
        try:
            client = self._get_chroma_client()
            kv_collection = client.get_or_create_collection("agent_memory")
            result = kv_collection.get(ids=[key])
            if result["documents"]:
                return result["documents"][0]
            return None
        except Exception as e:
            logger.warning("Memory recall failed for key '{}': {}", key, e)
            return None

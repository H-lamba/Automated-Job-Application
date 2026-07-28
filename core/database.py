"""
core/database.py — Async SQLAlchemy engine and session factory.

Usage in FastAPI routes:
    from core.database import get_db
    async def my_route(db: AsyncSession = Depends(get_db)):
        ...

Usage in standalone scripts / agents:
    async with get_session() as db:
        ...
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from core.logger import logger

# ──────────────────────────────────────────────────────────────────────────────
# Declarative base — all SQLAlchemy models inherit from this
# ──────────────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ──────────────────────────────────────────────────────────────────────────────
# Engine & session factory (initialised lazily on first call)
# ──────────────────────────────────────────────────────────────────────────────

_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_engine(database_url: str) -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            database_url,
            echo=False,           # Set True to log every SQL statement
            pool_pre_ping=True,   # Verify connections before use
            connect_args={"check_same_thread": False},  # Required for SQLite
        )
        logger.debug("Database engine created — url={}", database_url)
    return _engine


def _get_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    global _async_session_factory
    if _async_session_factory is None:
        engine = _get_engine(database_url)
        _async_session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Keep objects usable after commit
            autoflush=False,
        )
    return _async_session_factory


# ──────────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────────


async def init_db(database_url: str) -> None:
    """
    Create all tables defined by SQLAlchemy models.

    Called once at application startup. Safe to call multiple times
    (CREATE TABLE IF NOT EXISTS semantics).
    """
    # Import all models here so SQLAlchemy knows about them before create_all
    import models.application  # noqa: F401
    import models.job  # noqa: F401

    engine = _get_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialised")


@asynccontextmanager
async def get_session(database_url: str) -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that yields a database session.

    Commits on success, rolls back on any exception.
    """
    factory = _get_session_factory(database_url)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db(database_url: str) -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency generator — use with Depends().

    Note: FastAPI requires a plain async generator (not context manager).
    Wraps get_session() for compatibility.
    """
    factory = _get_session_factory(database_url)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_db() -> None:
    """Dispose the engine connection pool. Called at app shutdown."""
    global _engine, _async_session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
        logger.info("Database engine disposed")

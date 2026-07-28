"""
tests/conftest.py — Shared pytest fixtures.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.database import Base
from main import app

# ── In-memory SQLite for tests ────────────────────────────────────────────────

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide a rollback-isolated database session for each test."""
    factory = async_sessionmaker(bind=test_engine, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def sample_raw_job():
    """A minimal valid RawJob for testing."""
    from discovery.base_source import RawJob
    return RawJob(
        title="Software Engineer, AI",
        company="Anthropic",
        application_url="https://boards.greenhouse.io/anthropic/jobs/12345",
        source="greenhouse_api",
        source_job_id="12345",
        company_slug="anthropic",
        location="Remote",
        remote=True,
        description="We are building safe AI systems. Looking for engineers.",
        requirements="Python, ML, distributed systems",
    )


@pytest.fixture
def sample_profile():
    """Minimal user profile for testing."""
    from models.profile import (
        ContactInfo,
        Location,
        Preferences,
        Skill,
        Skills,
        UserProfile,
        WorkExperience,
    )
    return UserProfile(
        personal=ContactInfo(
            name="Himanshu",
            email="test@example.com",
            phone="+91 9999999999",
            location=Location(city="Bengaluru", country="India"),
        ),
        summary="Test summary",
        preferences=Preferences(
            target_roles=["Software Engineer", "ML Engineer"],
            remote="preferred",
            locations_ok=["Remote", "Bengaluru"],
        ),
        skills=Skills(
            technical=[
                Skill(name="Python", proficiency="expert"),
                Skill(name="Machine Learning", proficiency="advanced"),
            ]
        ),
        experience=[
            WorkExperience(
                company="Test Corp",
                title="Software Engineer",
                start_date="2022-01",
                current=True,
            )
        ],
    )

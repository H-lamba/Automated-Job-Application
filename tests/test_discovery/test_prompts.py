"""
tests/test_discovery/test_prompts.py — Unit tests for LLM prompt utilities.
"""

import pytest

from llm.prompts import keyword_pre_score


class TestKeywordPreScore:
    """Tests for the zero-cost keyword pre-scorer."""

    def test_exact_role_match_scores_high(self):
        score = keyword_pre_score(
            job_title="Software Engineer",
            job_description="Python and ML experience required.",
            target_roles=["Software Engineer"],
            technical_skills=["Python", "ML"],
        )
        assert score >= 60.0

    def test_no_match_scores_zero(self):
        score = keyword_pre_score(
            job_title="Marketing Manager",
            job_description="Running campaigns and brand awareness.",
            target_roles=["Software Engineer"],
            technical_skills=["Python", "FastAPI"],
        )
        assert score < 20.0

    def test_partial_skill_match(self):
        score = keyword_pre_score(
            job_title="Python Developer",
            job_description="Need Python and SQL skills.",
            target_roles=["Software Engineer"],
            technical_skills=["Python", "SQL", "Java", "Kubernetes"],
        )
        # 2 out of 4 skills matched
        assert 0 < score < 100

    def test_empty_skills_still_scores_title(self):
        score = keyword_pre_score(
            job_title="Software Engineer",
            job_description="",
            target_roles=["Software Engineer"],
            technical_skills=[],
        )
        assert score >= 30.0

    def test_score_capped_at_100(self):
        score = keyword_pre_score(
            job_title="Software Engineer ML",
            job_description="Python FastAPI ML Docker Kubernetes AWS",
            target_roles=["Software Engineer"],
            technical_skills=["Python", "FastAPI", "ML", "Docker"],
        )
        assert score <= 100.0

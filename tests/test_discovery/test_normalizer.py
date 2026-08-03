"""
tests/test_discovery/test_normalizer.py — Unit tests for the job normalizer.
"""


from discovery.normalizer import (
    JobNormalizer,
    compute_url_hash,
    detect_ats,
    detect_remote,
    strip_html,
)
from models.job import ATSType, JobStatus


def test_detect_ats_greenhouse():
    url = "https://boards.greenhouse.io/anthropic/jobs/12345"
    assert detect_ats(url) == ATSType.GREENHOUSE


def test_detect_ats_lever():
    url = "https://jobs.lever.co/notion/abc123"
    assert detect_ats(url) == ATSType.LEVER


def test_detect_ats_ashby():
    url = "https://jobs.ashbyhq.com/cursor/xyz"
    assert detect_ats(url) == ATSType.ASHBY


def test_detect_ats_unknown():
    url = "https://careers.somecompany.com/apply"
    assert detect_ats(url) == ATSType.UNKNOWN


def test_compute_url_hash_is_stable():
    url = "https://boards.greenhouse.io/test/jobs/1"
    h1 = compute_url_hash(url)
    h2 = compute_url_hash(url)
    assert h1 == h2
    assert len(h1) == 16


def test_compute_url_hash_different_urls():
    url1 = "https://boards.greenhouse.io/test/jobs/1"
    url2 = "https://boards.greenhouse.io/test/jobs/2"
    assert compute_url_hash(url1) != compute_url_hash(url2)


def test_strip_html_removes_tags():
    html = "<p>This is <b>bold</b> and <em>italic</em>.</p>"
    result = strip_html(html)
    assert "<p>" not in result
    assert "<b>" not in result
    assert "bold" in result
    assert "italic" in result


def test_strip_html_decodes_entities():
    html = "Company &amp; Co &lt;test&gt;"
    result = strip_html(html)
    assert "&amp;" not in result
    assert "Company & Co" in result


def test_detect_remote_from_flag(sample_raw_job):
    sample_raw_job.remote = True
    assert detect_remote(sample_raw_job) is True


def test_detect_remote_from_location(sample_raw_job):
    sample_raw_job.remote = False
    sample_raw_job.location = "Remote"
    assert detect_remote(sample_raw_job) is True


def test_detect_remote_not_remote(sample_raw_job):
    sample_raw_job.remote = False
    sample_raw_job.location = "New York, NY"
    sample_raw_job.title = "Software Engineer"
    sample_raw_job.description = "Office-based role in Manhattan."
    assert detect_remote(sample_raw_job) is False


class TestJobNormalizer:
    def setup_method(self):
        self.normalizer = JobNormalizer()

    def test_normalize_valid_job(self, sample_raw_job):
        job = self.normalizer.normalize(sample_raw_job)
        assert job is not None
        assert job.title == "Software Engineer, AI"
        assert job.company == "Anthropic"
        assert job.ats_type == ATSType.GREENHOUSE.value
        assert job.remote is True
        assert job.status == JobStatus.DISCOVERED.value
        assert job.relevance_score == 0.0
        assert len(job.url_hash) == 16

    def test_normalize_missing_title_returns_none(self, sample_raw_job):
        sample_raw_job.title = ""
        result = self.normalizer.normalize(sample_raw_job)
        assert result is None

    def test_normalize_missing_url_returns_none(self, sample_raw_job):
        sample_raw_job.application_url = ""
        result = self.normalizer.normalize(sample_raw_job)
        assert result is None

    def test_normalize_batch(self, sample_raw_job):
        jobs = self.normalizer.normalize_batch([sample_raw_job, sample_raw_job])
        assert len(jobs) == 2

    def test_normalize_strips_html_from_description(self, sample_raw_job):
        sample_raw_job.description = "<p>We are <b>building</b> AI.</p>"
        job = self.normalizer.normalize(sample_raw_job)
        assert job is not None
        assert "<p>" not in job.description
        assert "building" in job.description

"""
models/profile.py — User profile Pydantic models.

Not stored in the database — loaded from profile/profile.yaml at startup
and held in memory. Passed to agents that need it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Location(BaseModel):
    city: str = ""
    state: str = ""
    country: str = "India"


class ContactInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    first_name: str = ""
    last_name: str = ""
    email: str          # Use str for flexibility (EmailStr too strict)
    phone: str = ""
    location: Location = Location()
    linkedin: str = ""
    github: str = ""
    portfolio: str = ""
    website: str = ""


class Preferences(BaseModel):
    target_roles: list[str] = []
    industries: list[str] = []
    desired_salary_min: int | None = None
    desired_salary_max: int | None = None
    salary_currency: str = "INR"
    remote: Literal["required", "preferred", "ok_with_office", "no_preference"] = "preferred"
    locations_ok: list[str] = ["Remote"]
    willing_to_relocate: bool = False


class Skill(BaseModel):
    name: str
    proficiency: Literal["expert", "advanced", "intermediate", "beginner"] = "intermediate"
    years: int | None = None


class Skills(BaseModel):
    technical: list[Skill] = []
    soft: list[str] = []

    def technical_names(self) -> list[str]:
        """Return a flat list of technical skill names (for keyword matching)."""
        return [s.name for s in self.technical]


class WorkExperience(BaseModel):
    company: str
    title: str
    start_date: str         # e.g. "2022-06"
    end_date: str = ""      # Empty if current
    current: bool = False
    location: str = ""
    description: str = ""
    achievements: list[str] = []


class Education(BaseModel):
    institution: str
    degree: str
    field: str = ""
    start_date: str = ""
    end_date: str = ""
    gpa: float | None = None
    relevant_courses: list[str] = []


class Certification(BaseModel):
    name: str
    issuer: str = ""
    date: str = ""
    url: str = ""


class Language(BaseModel):
    language: str
    proficiency: Literal["native", "fluent", "professional", "conversational", "basic"] = "fluent"


# ──────────────────────────────────────────────────────────────────────────────
# Extra profile sections (from the enriched profile.yaml)
# ──────────────────────────────────────────────────────────────────────────────


class ResumeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    default_resume: str = ""
    variants: dict[str, str] = {}


class WorkAuthorization(BaseModel):
    model_config = ConfigDict(extra="ignore")
    country: str = "India"
    authorized: bool = True
    requires_visa_sponsorship: bool = False


class JobSearchConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    employment_types: list[str] = ["Full-time"]
    experience_levels: list[str] = ["Entry Level"]
    joining_availability: str = "Immediate"
    notice_period: str = "Immediate"
    apply_internationally: bool = False


class SearchKeywords(BaseModel):
    model_config = ConfigDict(extra="ignore")
    include: list[str] = []
    exclude: list[str] = []


class ApplicationPreferences(BaseModel):
    model_config = ConfigDict(extra="ignore")
    auto_apply: bool = False
    auto_fill_forms: bool = True
    auto_upload_resume: bool = True
    auto_generate_cover_letter: bool = True
    auto_answer_screening_questions: bool = True
    customize_resume_per_job: bool = False
    generate_followup_email: bool = False
    save_failed_applications: bool = True
    minimum_match_score: int = 75
    maximum_daily_applications: int = 50


class UserProfile(BaseModel):
    """
    Complete user profile loaded from profile.yaml.

    This is the single source of truth about the user that all agents
    use when filling forms, answering questions, or scoring relevance.
    """

    model_config = ConfigDict(extra="ignore")

    personal: ContactInfo
    summary: str = ""
    preferences: Preferences = Preferences()
    skills: Skills = Skills()
    experience: list[WorkExperience] = []
    education: list[Education] = []
    certifications: list[Certification] = []
    languages: list[Language] = [Language(language="English", proficiency="fluent")]

    # Extended fields from the enriched profile.yaml
    resume: ResumeConfig = ResumeConfig()
    work_authorization: WorkAuthorization = WorkAuthorization()
    job_search: JobSearchConfig = JobSearchConfig()
    search_keywords: SearchKeywords = SearchKeywords()
    application_preferences: ApplicationPreferences = ApplicationPreferences()
    preferred_company_tiers: list[str] = []
    blacklisted_companies: list[str] = []
    profiles: dict[str, str] = {}   # leetcode, kaggle, etc.
    documents: dict[str, Any] = {}

    # ── Derived properties useful for agents ────────────────────────────────

    def years_of_experience(self) -> float:
        """Approximate total years of professional experience."""
        from datetime import date

        total_months = 0
        for exp in self.experience:
            try:
                start = date.fromisoformat(exp.start_date + "-01")
                if exp.current or not exp.end_date:
                    end = date.today()
                else:
                    end = date.fromisoformat(exp.end_date + "-01")
                diff = (end.year - start.year) * 12 + (end.month - start.month)
                total_months += max(0, diff)
            except (ValueError, AttributeError):
                continue
        return round(total_months / 12, 1)

    def skills_summary(self) -> str:
        """Comma-separated skill names for quick LLM context injection."""
        names = self.skills.technical_names()
        names += self.skills.soft
        return ", ".join(names)

    def most_recent_title(self) -> str:
        """Return the title of the most recent job."""
        if not self.experience:
            return ""
        current = [e for e in self.experience if e.current]
        if current:
            return current[0].title
        return self.experience[0].title

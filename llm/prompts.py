"""
llm/prompts.py — All LLM prompt templates in one place.

Design principles:
- Templates are typed dataclasses, not raw f-strings scattered everywhere.
- Every prompt has a version tag so we can A/B test and roll back.
- Variables are named clearly and validated before formatting.
- Templates are never modified at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """Immutable prompt template with named placeholders."""

    name: str
    version: str
    system: str
    user_template: str

    def format(self, **kwargs) -> tuple[str, str]:
        """
        Return (system_prompt, user_message) with placeholders filled.

        Raises KeyError if a required placeholder is missing.
        """
        return self.system, self.user_template.format(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Job Relevance Scoring
# ──────────────────────────────────────────────────────────────────────────────

SCORE_JOB_PROMPT = PromptTemplate(
    name="score_job",
    version="2.0",
    system="""You are a precise job-fit evaluator for a software/ML engineer candidate.
Your job is to assess how well a job listing matches a candidate's technical profile.
You must respond ONLY with valid JSON. Do not include any explanation outside the JSON object.

CRITICAL RULES:
- If the job is primarily non-technical (accounting, legal, recruiting, sales, marketing,
  operations, HR, data-center physical work, policy), set title_match <= 20 and overall <= 30.
- Only score high if the job primarily requires software engineering, ML, data science,
  or closely related technical skills.
- "apply" must be true ONLY if overall >= 75 AND the role is clearly technical/engineering.""",
    user_template="""Evaluate this job against the candidate profile.

## Candidate Profile
- Name: {candidate_name}
- Current Title: {current_title}
- Years of Experience: {years_experience}
- Target Roles: {target_roles}
- Technical Skills: {technical_skills}
- Preferred Remote: {remote_preference}
- Preferred Locations: {preferred_locations}

## Job Listing
- Title: {job_title}
- Company: {company}
- Location: {location}
- Remote: {remote}
- Description:
{description}

## Scoring Criteria
Score each dimension from 0 to 100:
1. title_match: How closely the job title matches TARGET ROLES list (0 if non-technical)
2. skills_match: % of job requirements that match candidate's TECHNICAL SKILLS
3. experience_match: Does the required experience level match the candidate?
4. location_match: Does the location/remote policy fit preferences?
5. overall: Weighted score (title 30%, skills 40%, experience 20%, location 10%)
   • Non-technical roles (accountant, recruiter, legal, ops, sales, HR): overall <= 30
   • Partially technical (DevOps, TPM): overall <= 60
   • Fully technical (SWE, MLE, data science): can score up to 100

Also provide:
- "apply": true ONLY if overall >= 75 AND role is software/ML engineering
- "reasoning": 1-2 sentences on why this score
- "missing_skills": list of skills in job but absent from profile

Respond with this exact JSON structure:
{{
  "title_match": <int 0-100>,
  "skills_match": <int 0-100>,
  "experience_match": <int 0-100>,
  "location_match": <int 0-100>,
  "overall": <int 0-100>,
  "apply": <bool>,
  "reasoning": "<string>",
  "missing_skills": ["<skill>", ...]
}}""",
)

# ──────────────────────────────────────────────────────────────────────────────
# Application Question Answering
# ──────────────────────────────────────────────────────────────────────────────

ANSWER_QUESTION_PROMPT = PromptTemplate(
    name="answer_question",
    version="1.0",
    system="""You are helping a job applicant answer application questions authentically and professionally.
Your answers must be truthful, concise, and directly relevant to the question.
Respond ONLY with the answer text — no preamble, no quotes, no explanation.""",
    user_template="""Answer this application question on behalf of the candidate.

## Candidate Profile Summary
{profile_summary}

## Question
{question}

## Context
- Company: {company}
- Role: {job_title}
- Question Type: {question_type}
{options_context}

Write a {tone} answer that directly addresses the question.
Keep it under {max_words} words unless the question requires more detail.
Answer:""",
)

# ──────────────────────────────────────────────────────────────────────────────
# Planner (ReAct-style)
# ──────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """You are the Master Planner of an autonomous job application agent.
Your goal is to complete the user's objective by choosing the right tool at each step.

You have access to these tools:
{tools_list}

At each step you must respond with ONLY valid JSON in this format:
{{
  "thought": "<your reasoning about what to do next>",
  "tool": "<tool_name or 'done'>",
  "args": {{<tool arguments as key-value pairs>}},
  "final_answer": "<only present when tool is 'done'>"
}}

Rules:
- Always think before acting.
- Use 'done' when the objective is fully achieved.
- If a tool fails, try an alternative approach.
- Never call the same tool with the same arguments twice."""

PLANNER_USER_TEMPLATE = """Current objective: {objective}

Previous steps:
{history}

Current state:
{state}

What should you do next?"""

# ──────────────────────────────────────────────────────────────────────────────
# Vision — Screen Understanding
# ──────────────────────────────────────────────────────────────────────────────

VISION_UNDERSTAND_PROMPT = PromptTemplate(
    name="vision_understand",
    version="1.0",
    system="""You are a UI analyst that examines screenshots of web pages.
You identify form fields, buttons, error messages, and page state.
Respond ONLY with valid JSON.""",
    user_template="""Analyse this screenshot of a job application page.

Identify:
1. Page type (e.g., "form", "confirmation", "error", "login", "upload")
2. All visible form fields with their labels and types
3. Submit/Next button presence and label
4. Any error messages visible
5. Whether the application appears submitted/complete

Respond with this JSON structure:
{{
  "page_type": "<string>",
  "form_fields": [
    {{"label": "<string>", "type": "<text|email|phone|dropdown|checkbox|radio|file|textarea>", "required": <bool>}}
  ],
  "submit_button": {{"visible": <bool>, "label": "<string>"}},
  "errors": ["<error message>"],
  "is_complete": <bool>,
  "notes": "<any important observations>"
}}""",
)

# ──────────────────────────────────────────────────────────────────────────────
# Cover Letter Generation
# ──────────────────────────────────────────────────────────────────────────────

COVER_LETTER_PROMPT = PromptTemplate(
    name="cover_letter",
    version="1.0",
    system="""You are an expert technical writer helping a software professional write compelling cover letters.
Cover letters should be professional, specific to the role, and highlight genuine fit.
Never write generic filler. Be specific. Keep it under 350 words.""",
    user_template="""Write a cover letter for this application.

## Candidate
- Name: {candidate_name}
- Current Title: {current_title}
- Years of Experience: {years_experience}
- Top Skills: {top_skills}
- Key Achievements: {achievements}

## Target Role
- Title: {job_title}
- Company: {company}
- Job Description Highlights: {job_highlights}

Write a 3-paragraph cover letter:
1. Opening: Why this company and role specifically
2. Middle: Specific experience/achievements most relevant to this role
3. Closing: Call to action

Do NOT include date, address blocks, or "Dear Hiring Manager" — just the body paragraphs.""",
)


# ──────────────────────────────────────────────────────────────────────────────
# Keyword Pre-Filter (no LLM, just function)
# ──────────────────────────────────────────────────────────────────────────────

# Roles that are clearly non-engineering — skip before sending to LLM
_NON_TECH_TITLE_BLOCKLIST = [
    "accountant", "accounting", "recruiter", "recruiting", "talent",
    "legal", "counsel", "attorney", "compliance", "paralegal",
    "data center technician", "datacenter technician",
    "executive assistant", "office manager", "administrative",
    "sales", "account executive", "account manager",
    "marketing", "communications", "public relations",
    "finance", "financial analyst", "fp&a",
    "procurement", "sourcing", "supply chain",
    "policy", "government affairs", "regulatory",
    "human resources", "hr ", "people operations",
    "operations associate", "program manager",  # keep TPM / eng program manager
    "physician", "medical", "clinical",
]


def keyword_pre_score(
    job_title: str,
    job_description: str,
    target_roles: list[str],
    technical_skills: list[str],
) -> float:
    """
    Fast keyword-based pre-scoring (0.0–100.0).

    Used to filter obviously irrelevant jobs before calling the LLM.
    Returns a score; jobs below config.discovery.min_relevance_score
    are skipped without burning LLM tokens.
    """
    title_lower = job_title.lower()
    text = (job_title + " " + job_description).lower()
    score = 0.0

    # Hard blocklist: immediately discard non-engineering roles
    for blocked in _NON_TECH_TITLE_BLOCKLIST:
        if blocked in title_lower:
            return 0.0

    # Title match (up to 40 points)
    for role in target_roles:
        if role.lower() in text:
            score += 40.0
            break
        # Partial word match
        for word in role.lower().split():
            if len(word) > 3 and word in text:
                score += 15.0
                break

    # Skills match (up to 60 points)
    matched_skills = sum(1 for skill in technical_skills if skill.lower() in text)
    if technical_skills:
        skill_ratio = matched_skills / len(technical_skills)
        score += skill_ratio * 60.0

    return min(score, 100.0)

# ──────────────────────────────────────────────────────────────────────────────
# Vision & DOM Form Filling
# ──────────────────────────────────────────────────────────────────────────────

FORM_EXTRACTION_PROMPT = PromptTemplate(
    name="form_extraction",
    version="1.0",
    system="""You are an autonomous job application agent executing inside a browser.
Your task is to analyze an application form (provided as a screenshot and/or DOM inputs) and map the candidate's profile and resume data to the visible fields.
You must return a JSON object matching the `ExtractedForm` schema.
Instructions:
1. Identify all visible fields. For dropdown/radio, read the options.
2. Match them against the profile context. If a match is found, set `source="profile"`.
3. If the answer is found in the resume context, set `source="resume"`.
4. If the field is a custom screening question with no matching profile data (e.g. "Why do you want to work here?"), thoughtfully deduce or fabricate a professional, truthful-sounding answer consistent with the candidate's background. Set `source="fabricated"` and provide your `reasoning`.
5. NEVER fabricate legally sensitive facts (work authorization, criminal history, disability, veteran status, or salary if explicit range). For these, fall back to "Prefer not to answer", or the profile's desired salary, and mark `source="profile"`.
6. Use `previous_answers` to ensure consistency. If you fabricated an answer previously, reuse it.""",
    user_template="""Extract form data and generate answers for this application.

## Candidate Profile
Name: {name}
Email: {email}
Phone: {phone}
LinkedIn: {linkedin}
GitHub: {github}
Desired Salary: {salary}
Locations OK: {locations}

## Resume / Background Context
Top Skills: {skills}
Experience Summary: {experience}

## Previous Fabricated Answers (Use these if asked again)
{previous_answers}

## Job Context
Title: {job_title}
Company: {company}

## Scraped Form Inputs (Use `field_id` from here)
{form_inputs}

Return a JSON object conforming to this schema (do NOT wrap it in a markdown block, just output the JSON):
{{
  "page_type": "string",
  "fields": [
    {{
      "field_id": "string (the exact id/name from DOM)",
      "label": "string",
      "field_type": "string (text|dropdown|radio|checkbox|file)",
      "required": true/false,
      "options": ["string"],
      "answer": "string or null",
      "source": "profile|resume|fabricated|skipped",
      "confidence": 1.0,
      "reasoning": "string"
    }}
  ]
}}"""
)

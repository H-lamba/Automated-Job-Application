# AGENTS.md — Autonomous AI Career Agent

Compact guide for working in this repo. See `README.md` for the user-facing
overview; this file is for agents.

## Stack & entry points

- **Python 3.11+** (`.venv` was created with 3.14). Activate with `source .venv/bin/activate`.
- **FastAPI** server is `main.py` — uvicorn entry, lifespan wires DB / Ollama / scheduler.
- Standalone runner scripts at repo root (no FastAPI required):
  - `run_application.py` — Phase 2: applies to all `queued` jobs above `min_relevance_score`.
  - `run_discovery.py` — runs the full discovery + scoring pass end-to-end.
  - `run_sourcing_only.py [--source greenhouse|lever|ashby] [--out FILE]` — fetch only, no DB writes.
  - `rescore_jobs.py [LIMIT]` — re-LLM-score jobs already in the DB that have no score.
  - `inspect_ats_form.py <application_url>` — headed Playwright tool to verify form selectors (used during development only).
- `pyproject.toml` declares a console script: `career-agent` → `main:cli_app`.
- No `main:cli_app` symbol exists in `main.py`; the console script is a leftover
  from an earlier draft. **Don't try to invoke it** — run `python main.py` or
  `uvicorn main:app --reload` instead.

## Module map

| Path | Role |
|---|---|
| `agents/` | Orchestrators: `discovery_agent.py`, `application_agent.py`, `tailoring_agent.py`, `base_agent.py` |
| `api/` | FastAPI routers (`routes/{jobs,applications,documents}.py`) + Pydantic schemas |
| `browser/` | Playwright wrapper (`BrowserAgent`) |
| `core/` | `config.py`, `database.py` (async SQLAlchemy + aiosqlite), `exceptions.py`, `logger.py` |
| `discovery/` | Per-ATS fetchers (`greenhouse_api.py`, `lever_api.py`, `ashby_api.py`) + `normalizer.py` |
| `documents/` | Resume/cover-letter management + `FabricationStore` for screening-question answers |
| `llm/` | Unified LLM client (`client.py` for Ollama+MLX, `gemini_client.py`), prompts, response parser |
| `memory/` | Chroma-backed semantic memory + dedup |
| `models/` | SQLAlchemy ORM (`job.py`, `application.py`, `profile.py`) |
| `profile/` | YAML profile loader + `profile.yaml` (Himanshu's actual profile, lives here) |
| `scheduler/` | APScheduler config; discovery every Nh, applications at cron `30 3 * * 1-5` |
| `vision/` | Screenshot → VLM → form-field extraction |
| `data/` | Runtime artefacts (DB, Chroma, screenshots, logs, tailored docs, fabricated answers) — gitignored |

## Dev commands

```bash
source .venv/bin/activate          # required — Ollama/MLX look for keys/env in shell
pip install -e ".[dev]"            # installs pytest, ruff, mypy, respx, factory-boy
python main.py                     # FastAPI on :8000, /docs at /docs
uvicorn main:app --reload          # equivalent

# Tests
pytest                                  # all tests (asyncio_mode = auto)
pytest tests/test_discovery             # one folder
pytest -k keyword_pre_score             # single test by name
pytest --lf                            # re-run last failures

# Lint / typecheck (declared in pyproject under [tool.ruff] and [tool.mypy])
ruff check .
ruff format .
mypy .
```

There is **no CI, no pre-commit, no Makefile, no task runner**. Run the commands
above directly.

## Critical gotchas

- **`application.dry_run` defaults to `true`** in `config.yaml`. Applications
  walk the full pipeline but never click Submit. Set `dry_run: false` only when
  intentionally going live. The README echoes this — don't ship code that
  assumes submissions actually fire.
- **Ollama must be running** (`ollama serve`) before any scoring/discovery path.
  The lifespan in `main.py:62` logs a warning and continues; the standalone
  runners `run_discovery.py` and `rescore_jobs.py` `sys.exit(1)` on failure.
  Configured base URL is `http://127.0.0.1:11434`; override via
  `OLLAMA_BASE_URL` env or `llm.ollama_base_url` in config.
- **LLM provider switch** is `config.yaml` → `llm.provider`: `ollama` or `gemini`.
  When `gemini`, set `GEMINI_API_KEY` in `.env`. Vision backend is a separate
  switch: `llm.vision_backend` ∈ {`ollama`, `mlx`}. MLX is Apple-Silicon-only.
- **Ashby quirk** (see comment in `config.yaml:107` and `discovery/ashby_api.py:65`):
  Ashby's public board uses **GET, not POST** — POST returns 401.
- **`config.yaml` vs `.env`**: YAML is the base layer; env vars override via the
  `SECTION__KEY` convention (e.g. `LLM__REASONING_MODEL=qwen3:8b`). See
  `core/config.py:117-122`. `.env` is gitignored.
- **`storage.documents_dir` is an absolute path** in `config.yaml`
  (`/Users/himanshu/Desktop/Working/Linkdin`). Changing this requires editing
  the YAML, not just env vars — env override of nested list-of-paths is
  awkward.
- **`run_application.py` reads `status == 'queued'`** in
  `agents/application_agent.py:46` — strings, not the `JobStatus` enum used
  elsewhere. If you add status states, keep the lowercase string the agent
  queries.
- **`ApplicationAgent.__init__` is sync but lazily inits a Playwright browser
  inside `process_queue`.** Instantiating it does not need a running display;
  applying does (headless is configurable in `config.yaml`, defaults to `false`
  so you can watch).
- **Tests use in-memory SQLite** (`tests/conftest.py:21`) and `session`-scoped
  engine. The `event_loop` fixture is also `session`-scoped; mixing scopes with
  pytest-asyncio will break. New fixtures should respect the same scope.
- **Logging goes to `data/logs/`** with JSON rotation (`loguru`, 50 MB × 30 days).
  Console level is set in `config.yaml` → `logging.level`.

## Workflow conventions

- **Style**: ruff (`E,F,I,UP,B,SIM`, line-length 100, target py311) + mypy
  (`strict = false`, `ignore_missing_imports = true`). No formatter is enforced
  beyond `ruff format`.
- **Profile is checked in** (`profile/profile.yaml` contains real PII — email,
  phone, LinkedIn, GitHub). Treat it as personal data, not a template.
- **`raw_jobs.json` is gitignored** — it's a dump from `run_sourcing_only.py`.
- **`data/career_agent.db` is gitignored** — never commit DB files.
- **Adding an ATS source**: implement `JobSource` from `discovery/base_source.py`,
  register a list of slugs in `config.yaml` (`<ats>_companies`), and wire the
  source into `agents/discovery_agent.py` and `run_sourcing_only.py`.
- **Adding a screening-question answerer**: append to `documents/fabrication_store.py`
  and ensure `application.fabrication_review_required` is respected if you want
  human-in-the-loop.

## Testing notes

- Only `tests/test_discovery/` exists; coverage is thin (normalizer + a prompt
  util). Most of `agents/`, `llm/`, and `browser/` have no tests.
- Tests are pure unit tests — no live network, no live Ollama. If you add a test
  that hits the network, mock with `respx` (already in dev deps).
- `pytest-asyncio` is in `auto` mode — async test functions don't need a marker.
- The conftest imports `from main import app` and `Base`; any model added to
  `models/` must be imported inside `core/database.init_db` to register with
  `Base.metadata.create_all` (see `core/database.py:85-86`).

## When changing things

- LLM prompts live in `llm/prompts.py` — keep `keyword_pre_score` and
  `SCORE_JOB_PROMPT` in sync with the response schema in `llm/response_parser.py`.
- Config schema lives in `core/config.py` (Pydantic models per section). New
  config keys need both a model field and a default in `config.yaml`.
- New scheduled work goes in `scheduler/job_scheduler.py`. The existing
  AsyncIOScheduler is shared with FastAPI's event loop.
- Any code that touches a real ATS form (browser automation, file uploads) must
  be verified against `inspect_ats_form.py` selectors before shipping.

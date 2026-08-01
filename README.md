<div align="center">
  <img src="https://raw.githubusercontent.com/H-lamba/Automated-Job-Application/main/.github/logo.png" width="120" alt="Career Agent Logo" />
  <h1>🤖 Autonomous AI Career Agent</h1>
  <p><em>An end-to-end autonomous agent that discovers jobs, scores them against your profile, and applies automatically using a vision-backed web automation engine.</em></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version" />
    <img src="https://img.shields.io/badge/Playwright-Automation-green.svg" alt="Playwright" />
    <img src="https://img.shields.io/badge/LLM-Gemini_&_Ollama-orange.svg" alt="LLM" />
  </p>
</div>

---

## 🌟 Project Architecture

Our agent follows a modular architecture composed of three main layers: **Intelligence**, **Automation**, and **Storage**. 

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4f46e5', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#3730a3', 'lineColor': '#818cf8', 'tertiaryColor': '#e0e7ff', 'edgeLabelBackground': '#f3f4f6'}}}%%
graph TD
    classDef llm fill:#8b5cf6,stroke:#5b21b6,color:white,stroke-width:2px;
    classDef browser fill:#10b981,stroke:#047857,color:white,stroke-width:2px;
    classDef db fill:#f59e0b,stroke:#b45309,color:white,stroke-width:2px;
    classDef core fill:#3b82f6,stroke:#1d4ed8,color:white,stroke-width:2px;
    
    subgraph Core
        C[Profile Loader]:::core --> |Parses YAML| Config[Config Manager]:::core
    end
    
    subgraph Discovery Phase
        JE[Job Extractors]:::browser --> |Scrapes ATS| DB[(SQLite Database)]:::db
        LLM1[Gemini 3.5 Flash / Ollama]:::llm --> |Scores Match %| DB
    end
    
    subgraph Application Phase
        DB --> |Queues jobs| AA[Application Agent]:::core
        AA --> BA[Browser Agent (Playwright)]:::browser
        BA --> |Takes Screenshots| VM[Vision Module]:::llm
        VM --> |Form Recognition| AA
        AA --> |Fills Form & Submits| BA
    end
```

---

## 🚀 Development Phases & Achievements

### 🎯 Phase 1: Foundation & Discovery Engine
*The infrastructure to find jobs and match them with high precision.*

✅ **YAML User Profile:** Structured ATS-friendly schema for skills, experiences, and target jobs.
✅ **ATS Extractors:** Reverse-engineered APIs for **Greenhouse, Lever, and Ashby** to pull structured jobs.
✅ **AI Job Scoring:** LLM integration (Gemini/Ollama) to semantically score a job description against the user's profile, generating a 0-100 `relevance_score`.
✅ **Database Persistence:** SQLite + SQLAlchemy ORM handling deduplication, history, and status queues.

### 🕹️ Phase 2: Autonomous Application Engine (Current)
*Giving the agent the ability to act on the web autonomously.*

✅ **Playwright Headless Browser:** Full automation wrapper with screenshot capabilities and session management.
✅ **Vision-Language Model (VLM):** Deep integration with `gemini-3.5-flash` to take visual screenshots of the DOM and answer: *"Is this an application form?"*
✅ **Adaptive Form Filling:** DOM-selector algorithms targeting dynamic input fields (name, email, phone, LinkedIn, GitHub) via unified matching.
✅ **LLM Factory Pattern:** Seamlessly switch between Local (Ollama) and Cloud API (Gemini) backends through `config.yaml`.
✅ **Automated Submit:** Live integration mapping that successfully navigates to an ATS and hits the submit button.

---

## ⚙️ Configuration

The entire system behavior is controlled by `config.yaml`. You can choose your intelligence provider seamlessly:

```yaml
llm:
  provider: "gemini"    # Switch to "ollama" for local execution!
  reasoning_model: "gemini-3.5-flash"
  vision_model: "gemini-3.5-flash"
```

## 🛠️ Usage

To kick off an autonomous application sprint:
```bash
python run_application.py
```
*Tip: Ensure your `GEMINI_API_KEY` is loaded in your `.env` file, and `dry_run: false` is set in the config to trigger real submissions.*

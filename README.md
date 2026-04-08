---
title: metaHackathon
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# AI AutoGen QA — Automated Test Case Generator

> **OpenEnv Hackathon Submission** — An AI-powered QA environment that transforms software documentation into comprehensive, structured test suites using a reflection-augmented LangGraph pipeline.

## What It Does

**AI AutoGen QA** solves a real-world problem every software team faces: generating thorough test coverage from design documents. Given Feature Requirement Documents (FRDs), Technical Design docs, and API specifications, it automatically produces:

- **10-15 high-level test scenarios** covering functional, security, edge-case, and integration dimensions
- **2-3 detailed test cases per scenario** with steps, preconditions, expected results, request/response schemas
- **Quality-validated output** through a reflection stage that catches coverage gaps and enforces completeness

The environment is exposed as a standard **OpenEnv HTTP API** (`reset` / `step` / `state`) so any RL agent or evaluation harness can interact with it programmatically.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  inference.py (OpenAI Client)                               │
│  Generates docs via LLM → sends to env → logs results      │
└───────────────┬─────────────────────────────────────────────┘
                │  POST /reset, POST /step, GET /state
                ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI Server (server.py)                                 │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         LangGraph 4-Node Pipeline                      │ │
│  │                                                        │ │
│  │  ┌──────────────┐    ┌──────────────┐                  │ │
│  │  │ assist_stage1 │───▶│reflect_stage1│──┐               │ │
│  │  │ (scenarios)   │◀───│ (validate)   │  │ if finished   │ │
│  │  └──────────────┘    └──────────────┘  │               │ │
│  │         ▲ loop if not done              ▼               │ │
│  │  ┌──────────────┐    ┌──────────────┐                  │ │
│  │  │ assist_stage2 │───▶│reflect_stage2│                  │ │
│  │  │ (test cases)  │    │ (quality)    │                  │ │
│  │  └──────────────┘    └──────────────┘                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  Reward ∈ [0.0, 1.0] based on task-specific grader          │
└─────────────────────────────────────────────────────────────┘
```

## OpenEnv API

| Endpoint  | Method | Description                                            |
|-----------|--------|--------------------------------------------------------|
| `/reset`  | POST   | Reset environment; returns initial observation         |
| `/step`   | POST   | Accepts `QAAction` with doc texts; runs pipeline       |
| `/state`  | GET    | Returns current `QAState`                              |
| `/health` | GET    | Liveness probe (`{"status": "healthy"}`)               |

## Typed Models (`models.py`)

| Model            | Key Fields                                                                 |
|------------------|---------------------------------------------------------------------------|
| **QAAction**     | `task_id` (easy/medium/hard), `design_doc_texts`, `api_doc_text`          |
| **QAObservation**| `done`, `reward` (0.0–1.0), `scenarios_generated`, `test_cases_generated`, `quality_score` |
| **QAState**      | `episode_id`, `step_count`, `task_id`, `stage`, completion flags          |

## Tasks & Grading

| Task       | Input              | What It Measures                                    | Scoring                                          |
|------------|--------------------|----------------------------------------------------|--------------------------------------------------|
| **easy**   | 1 API doc          | Scenario generation from a single source           | `min(count / 5, 1.0)`                            |
| **medium** | FRD + API doc      | Multi-doc coverage + reflection quality            | 60% scenario count + 40% reflection pass         |
| **hard**   | FRD + API doc      | Full pipeline: scenarios, cases, quality           | 25% scenarios + 15% S1 + 25% cases + 10% S2 + 25% quality |

## Project Structure

```
├── inference.py          # Baseline inference (OpenAI Client + OpenEnv API)
├── server.py             # FastAPI OpenEnv server (reset/step/state/health)
├── models.py             # Typed Pydantic models (Action/Observation/State)
├── openenv.yaml          # OpenEnv manifest
├── Dockerfile            # HuggingFace Spaces deployment
├── requirements.txt      # Pinned dependencies
├── .env.example          # Environment variable template
├── graders/              # Task graders (easy/medium/hard)
│   ├── easy.py
│   ├── medium.py
│   └── hard.py
└── src/
    ├── app.py            # Streamlit UI (interactive interface)
    ├── constants.py      # Centralized config (loads .env)
    └── qa_agent/
        ├── pdf_graph_agent.py         # LangGraph state machine (4 nodes)
        ├── assistant_thread_manager.py # OpenAI Chat Completions wrapper
        └── prompts/
            └── pdf_graph_prompts.py   # Prompt templates for each stage
```

## Environment Variables

| Variable       | Description                          | Default                       |
|----------------|--------------------------------------|-------------------------------|
| `API_BASE_URL` | LLM API endpoint                     | `https://api.openai.com/v1`   |
| `MODEL_NAME`   | Model identifier                     | `gpt-4o`                      |
| `HF_TOKEN`     | API key (OpenAI / HuggingFace)       | —                             |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your API key
cp .env.example .env
# Edit .env with your actual HF_TOKEN / API key

# 3. Start the OpenEnv server
uvicorn server:app --host 0.0.0.0 --port 8000

# 4. In another terminal, run inference
python inference.py
```

## Docker

```bash
docker build -t ai-autogen-qa .
docker run -p 8000:8000 \
  -e HF_TOKEN="your-key" \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o" \
  ai-autogen-qa
```

## How the Pipeline Works

1. **Stage 1 — Scenario Generation**: The LLM analyzes uploaded documents and produces 10-15 high-level test scenarios using advanced test design techniques (boundary value analysis, equivalence partitioning, state transition testing).

2. **Stage 1 — Reflection**: A QA reflection pass validates scenario completeness, checks for gaps in functional/security/edge-case coverage, and requests revisions if needed (up to 3 iterations).

3. **Stage 2 — Test Case Generation**: Each validated scenario is expanded into 2-3 detailed test cases with structured fields: steps, preconditions, test data, expected results, request/response bodies.

4. **Stage 2 — Quality Assessment**: A final quality check evaluates test case completeness (required fields, meaningful content, step count, API structure) and produces a 0.0-1.0 quality score.

## Reward Design

Rewards provide **partial progress signals** at every stage:

- Even generating 1 scenario yields a non-zero reward
- Reflection completion adds a bonus
- Test case generation scales linearly
- Quality score provides fine-grained feedback

This enables RL agents to learn incrementally rather than receiving only binary pass/fail.

## Tech Stack

- **FastAPI + Uvicorn** — OpenEnv HTTP server
- **LangGraph** — State machine orchestration with memory and conditional edges
- **OpenAI Chat Completions** — LLM-powered generation (compatible with any OpenAI-compatible API)
- **Pydantic v2** — Typed models with validation
- **Streamlit** — Interactive web UI for manual testing

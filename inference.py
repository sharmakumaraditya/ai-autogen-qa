#!/usr/bin/env python3
"""
inference.py — Hackathon baseline inference script for AI AutoGen QA.

Talks to the OpenEnv server (POST /reset, POST /step, GET /state)
and uses the OpenAI Client for LLM calls.

Emits structured [START], [STEP], [END] logs as required.

Required env vars:
    API_BASE_URL   – LLM API endpoint  (default: https://api.openai.com/v1)
    MODEL_NAME     – model identifier   (default: gpt-4o)
    HF_TOKEN       – API key
"""

import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import requests
from openai import OpenAI

ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

MAX_RETRIES = 3
RETRY_DELAY = 5
REQUEST_TIMEOUT = 300


def log(tag: str, payload: dict):
    print(f"[{tag}] {json.dumps(payload)}", flush=True)


def _request_with_retry(method: str, url: str, **kwargs):
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise
        except requests.exceptions.HTTPError as e:
            if attempt < MAX_RETRIES and resp.status_code >= 500:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


def call_reset(episode_id: str = "") -> dict:
    return _request_with_retry("POST", f"{ENV_URL}/reset", json={"episode_id": episode_id})


def call_step(action: dict) -> dict:
    return _request_with_retry("POST", f"{ENV_URL}/step", json={"action": action})


def call_state() -> dict:
    return _request_with_retry("GET", f"{ENV_URL}/state")


def generate_doc_text(description: str) -> str:
    """Use OpenAI Client to generate a sample document for testing."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a technical writer. Write concise, structured documentation."},
                    {"role": "user", "content": description},
                ],
                max_tokens=1200,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


def run_task(task_id: str):
    """Run one task through the OpenEnv endpoints."""

    log("START", {
        "task_id": task_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    # 1. Reset
    reset_result = call_reset(episode_id=f"inference_{task_id}")
    log("STEP", {
        "task_id": task_id,
        "step": 0,
        "node": "reset",
        "observation": reset_result.get("observation", {}),
    })

    # 2. Generate documents via OpenAI Client
    log("STEP", {
        "task_id": task_id,
        "step": 1,
        "node": "generate_docs",
        "message": "Generating sample documents via OpenAI Client",
    })

    api_doc_text = generate_doc_text(
        "Write a detailed API documentation for a User Management REST API with "
        "endpoints: POST /users, GET /users, GET /users/{id}, PUT /users/{id}, "
        "DELETE /users/{id}, POST /auth/login. Include request/response JSON schemas, "
        "HTTP status codes, authentication requirements (JWT Bearer tokens), "
        "rate limiting rules, and pagination parameters."
    )

    design_doc_texts = []
    if task_id in ("medium", "hard"):
        frd_text = generate_doc_text(
            "Write a Feature Requirement Document (FRD) for a User Management system. "
            "Include detailed requirements for: user CRUD operations, role-based access "
            "control (admin/editor/viewer), JWT authentication with refresh tokens, "
            "email validation, password policy (min 8 chars, uppercase, special char), "
            "cursor-based pagination, soft-delete with 30-day retention, and audit logging "
            "with actor/action/timestamp/resource fields."
        )
        design_doc_texts.append(frd_text)

    log("STEP", {
        "task_id": task_id,
        "step": 2,
        "node": "docs_ready",
        "design_docs": len(design_doc_texts),
        "api_doc_length": len(api_doc_text),
    })

    # 3. Step: run the QA pipeline
    action = {
        "task_id": task_id,
        "design_doc_texts": design_doc_texts,
        "api_doc_text": api_doc_text,
    }

    step_result = call_step(action)
    obs = step_result.get("observation", {})
    reward = step_result.get("reward", 0.0)
    done = step_result.get("done", True)

    log("STEP", {
        "task_id": task_id,
        "step": 3,
        "node": "pipeline_complete",
        "scenarios": obs.get("scenarios_generated", 0),
        "test_cases": obs.get("test_cases_generated", 0),
        "quality_score": obs.get("quality_score", 0.0),
        "reward": reward,
        "done": done,
    })

    # 4. Fetch final state
    state = call_state()
    log("STEP", {
        "task_id": task_id,
        "step": 4,
        "node": "get_state",
        "state": state,
    })

    log("END", {
        "task_id": task_id,
        "reward": reward,
        "scenarios_generated": obs.get("scenarios_generated", 0),
        "test_cases_generated": obs.get("test_cases_generated", 0),
        "quality_score": obs.get("quality_score", 0.0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    return reward


def main():
    tasks = ["easy", "medium", "hard"]
    results = {}

    for task_id in tasks:
        print(f"\n{'=' * 60}")
        print(f"  TASK: {task_id}")
        print(f"{'=' * 60}\n")
        try:
            score = run_task(task_id)
        except Exception as e:
            print(f"  ERROR running task {task_id}: {e}", file=sys.stderr)
            score = 0.0
            log("END", {
                "task_id": task_id,
                "reward": 0.0,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        results[task_id] = score

    print(f"\n{'=' * 60}")
    print("  FINAL SCORES")
    print(f"{'=' * 60}")
    for tid, s in results.items():
        print(f"  {tid:8s}: {s}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

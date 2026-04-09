#!/usr/bin/env python3
"""
inference.py — Hackathon baseline inference script for AI AutoGen QA.

Talks to the OpenEnv server (POST /reset, POST /step, GET /state)
and uses the OpenAI Client for LLM calls.

Emits structured [START], [STEP], [END] logs as required by the sample format.

Required env vars:
    API_BASE_URL   – LLM API endpoint  (default: https://api.openai.com/v1)
    MODEL_NAME     – model identifier   (default: gpt-4o)
    HF_TOKEN       – API key
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

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
BENCHMARK = "ai-autogen-qa"


def _clamp_score(v: float) -> float:
    """Clamp score to strictly between 0 and 1 (exclusive)."""
    return max(0.01, min(float(v), 0.99))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


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


def run_task(task_id: str) -> float:
    """Run one task through the OpenEnv endpoints."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01  # default to valid score

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Step 1: Reset
        reset_result = call_reset(episode_id=f"inference_{task_id}")
        steps_taken = 1
        log_step(step=1, action="reset", reward=0.01, done=False)
        rewards.append(0.01)

        # Step 2: Generate documents via OpenAI Client
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

        steps_taken = 2
        log_step(step=2, action="generate_docs", reward=0.01, done=False)
        rewards.append(0.01)

        # Step 3: Run the QA pipeline
        action = {
            "task_id": task_id,
            "design_doc_texts": design_doc_texts,
            "api_doc_text": api_doc_text,
        }

        step_result = call_step(action)
        obs = step_result.get("observation", {})
        reward = _clamp_score(step_result.get("reward", 0.01))
        done = step_result.get("done", True)

        steps_taken = 3
        log_step(step=3, action="run_pipeline", reward=reward, done=done)
        rewards.append(reward)

        # Step 4: Fetch final state
        state = call_state()
        steps_taken = 4
        log_step(step=4, action="get_state", reward=reward, done=True)
        rewards.append(reward)

        score = reward

    except Exception as e:
        print(f"[DEBUG] Error in task {task_id}: {e}", file=sys.stderr, flush=True)
        score = 0.01
        if not rewards:
            rewards.append(0.01)

    success = score > 0.5
    log_end(success=success, steps=steps_taken, score=_clamp_score(score), rewards=[_clamp_score(r) for r in rewards])

    return score


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
            score = 0.01
            log_end(success=False, steps=0, score=0.01, rewards=[0.01])
        results[task_id] = score

    print(f"\n{'=' * 60}")
    print("  FINAL SCORES")
    print(f"{'=' * 60}")
    for tid, s in results.items():
        print(f"  {tid:8s}: {s}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

"""
OpenEnv-compatible FastAPI server for AI AutoGen QA environment.

Endpoints:
    POST /reset   — reset environment, return initial observation
    POST /step    — run the QA pipeline, return observation with reward
    GET  /state   — return current internal state
    GET  /health  — liveness probe
"""

import os
import sys
import uuid
import tempfile
import traceback
from typing import Any, Dict

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import QAAction, QAObservation, QAState

app = FastAPI(title="AI AutoGen QA — OpenEnv Environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = QAState()


def _score(
    task_id: str,
    scenarios: list,
    test_cases: list,
    quality: float,
    finished_s1: bool,
    finished_s2: bool,
) -> float:
    """Compute reward in [0.0, 1.0] matching graders."""
    if task_id == "easy":
        return min(round(len(scenarios) / 5, 2), 1.0)

    if task_id == "medium":
        s = 0.0
        c = len(scenarios)
        if c >= 10:
            s += 0.6
        elif c >= 5:
            s += round(0.6 * c / 10, 2)
        if finished_s1:
            s += 0.4
        return min(round(s, 2), 1.0)

    # hard
    s = 0.0
    if len(scenarios) >= 10:
        s += 0.25
    elif len(scenarios) >= 5:
        s += round(0.25 * len(scenarios) / 10, 2)
    if finished_s1:
        s += 0.15
    if len(test_cases) >= 10:
        s += 0.25
    elif test_cases:
        s += round(0.25 * len(test_cases) / 10, 2)
    if finished_s2:
        s += 0.10
    s += round(0.25 * min(quality, 1.0), 2)
    return min(round(s, 2), 1.0)


@app.post("/reset")
async def reset(request: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    global _state
    _state = QAState(
        episode_id=request.get("episode_id") or str(uuid.uuid4()),
        step_count=0,
        task_id="",
        stage="ready",
    )
    obs = QAObservation(
        done=False,
        reward=0.0,
        stage="ready",
        message="Environment reset. Send a step() with task_id and document texts.",
    )
    return {
        "observation": obs.model_dump(exclude={"done", "reward", "metadata"}),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.post("/step")
async def step(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    global _state

    action_data = request.get("action", request)
    try:
        action = QAAction(**action_data)
    except Exception as e:
        obs = QAObservation(
            done=True,
            reward=0.0,
            stage="error",
            message=f"Invalid action: {e}",
        )
        return {
            "observation": obs.model_dump(exclude={"done", "reward", "metadata"}),
            "reward": obs.reward,
            "done": obs.done,
        }

    _state.step_count += 1
    _state.task_id = action.task_id

    try:
        _state.stage = "running_pipeline"

        tmpdir = tempfile.mkdtemp(prefix="qa_env_")
        design_paths = []
        for i, text in enumerate(action.design_doc_texts):
            p = os.path.join(tmpdir, f"design_{i}.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            design_paths.append(p)

        api_path = ""
        if action.api_doc_text:
            api_path = os.path.join(tmpdir, "api_doc.txt")
            with open(api_path, "w", encoding="utf-8") as f:
                f.write(action.api_doc_text)

        from qa_agent.pdf_graph_agent import PDFGraph

        pdf_graph = PDFGraph()
        graph = pdf_graph.get_memory_graph()

        inputs = {
            "input": "",
            "target_app": "PDF Processing",
            "design_documents": [f"design_{i}.txt" for i in range(len(design_paths))],
            "api_document": "api_doc.txt" if api_path else "",
            "design_file_paths": design_paths,
            "api_file_path": api_path,
            "temp_files": [],
            "message_history": [],
            "test_list": [],
            "is_scenario_list_processed": False,
            "scenario_list": [],
            "current_scenario": (0, {}),
            "current_test": (0, ""),
            "current_test_details": [],
            "test_details_list": [],
            "is_test_list_processed": False,
            "question": "",
            "stage1_thread_id": "",
            "stage1_revisions": 0,
            "is_finished_stage1": False,
            "stage2_thread_id": "",
            "stage2_revisions": 0,
            "is_finished_stage2": False,
            "stage3_thread_id": "",
            "stage3_revisions": 0,
            "is_finished_stage3": False,
            "processed_scenarios": [],
        }

        thread_config = {"configurable": {"thread_id": _state.episode_id or "env"}}
        final = {}
        for output in graph.stream(inputs, thread_config):
            node = list(output.keys())[0]
            final = output[node]

        scenarios = final.get("scenario_list", [])
        test_cases = final.get("test_details_list", [])
        quality = final.get("stage2_quality_score", 0.0)
        if quality is None:
            quality = 0.0
        finished_s1 = final.get("is_finished_stage1", False)
        finished_s2 = final.get("is_finished_stage2", False)

        reward = _score(action.task_id, scenarios, test_cases, quality, finished_s1, finished_s2)

        _state.stage = "done"
        _state.scenarios_generated = len(scenarios)
        _state.test_cases_generated = len(test_cases)
        _state.quality_score = quality
        _state.is_finished_stage1 = finished_s1
        _state.is_finished_stage2 = finished_s2

        obs = QAObservation(
            done=True,
            reward=reward,
            scenarios_generated=len(scenarios),
            test_cases_generated=len(test_cases),
            quality_score=quality,
            stage="done",
            message=f"Pipeline complete. {len(scenarios)} scenarios, {len(test_cases)} test cases. Reward: {reward}",
            scenarios=[s[1] if isinstance(s, tuple) else s for s in scenarios],
            test_cases=test_cases[:5],
        )

    except Exception as e:
        traceback.print_exc()
        _state.stage = "error"
        obs = QAObservation(
            done=True,
            reward=0.0,
            stage="error",
            message=f"Pipeline error: {str(e)}",
        )

    return {
        "observation": obs.model_dump(exclude={"done", "reward", "metadata"}),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    return _state.model_dump()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    main()

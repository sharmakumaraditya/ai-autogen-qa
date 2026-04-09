"""Typed Pydantic models for the AI AutoGen QA environment.

Defines Action, Observation, and State following the OpenEnv spec:
  - Action:      what the agent sends each step
  - Observation:  what comes back (done flag + reward + payload)
  - State:        internal environment state (inspectable via GET /state)
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class QAAction(BaseModel):
    """Agent action: submit document texts and trigger the QA pipeline."""
    task_id: str = Field(
        "easy",
        description="Task difficulty level: easy | medium | hard",
    )
    design_doc_texts: List[str] = Field(
        default_factory=list,
        description="Contents of design/FRD documents (plain text)",
    )
    api_doc_text: str = Field(
        "",
        description="Contents of the API documentation (plain text)",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("easy", "medium", "hard"):
            raise ValueError(f"task_id must be easy, medium, or hard — got '{v}'")
        return v


class QAObservation(BaseModel):
    """Observation returned after reset() or step()."""
    done: bool = False
    reward: Optional[float] = Field(None, gt=0.0, lt=1.0)
    scenarios_generated: int = Field(0, ge=0)
    test_cases_generated: int = Field(0, ge=0)
    quality_score: float = Field(0.01, ge=0.0, le=1.0)
    stage: str = "idle"
    message: str = ""
    scenarios: List[Dict[str, Any]] = Field(default_factory=list)
    test_cases: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QAState(BaseModel):
    """Internal environment state exposed via GET /state."""
    episode_id: Optional[str] = None
    step_count: int = Field(0, ge=0)
    task_id: str = ""
    stage: str = "idle"
    scenarios_generated: int = Field(0, ge=0)
    test_cases_generated: int = Field(0, ge=0)
    quality_score: float = Field(0.01, ge=0.0, le=1.0)
    is_finished_stage1: bool = False
    is_finished_stage2: bool = False

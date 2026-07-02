# src/schema.py
from __future__ import annotations
import json
from pydantic import BaseModel, ValidationError


class PersonRecord(BaseModel):
    """The strict JSON schema the model must produce."""
    name: str
    age: int
    job_title: str
    company: str
    city: str
    email: str

    model_config = {"extra": "forbid"}  # reject unknown keys

    def to_minified_json(self) -> str:
        # compact, sorted keys → deterministic targets for training
        return json.dumps(self.model_dump(), separators=(",", ":"), sort_keys=True)


def is_valid_record(text: str) -> bool:
    """True if `text` parses to JSON AND matches the schema exactly."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    try:
        PersonRecord(**obj)
        return True
    except (ValidationError, TypeError):
        return False


def safe_parse(text: str) -> dict | None:
    """Return dict if parseable JSON object, else None (schema not enforced)."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None

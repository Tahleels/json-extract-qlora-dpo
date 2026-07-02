# src/data/corrupt.py
from __future__ import annotations
import json, random


def _fenced(clean: str) -> str:
    return f"```json\n{clean}\n```"

def _preamble(clean: str) -> str:
    lead = random.choice([
        "Sure! Here is the extracted JSON:\n",
        "Here you go:\n",
        "The extracted information is:\n",
    ])
    return lead + clean

def _trailing_comma(clean: str) -> str:
    return clean[:-1] + ",}" if clean.endswith("}") else clean

def _single_quotes(clean: str) -> str:
    return clean.replace('"', "'")

def _drop_field(clean: str) -> str:
    obj = json.loads(clean)
    if len(obj) > 1:
        obj.pop(random.choice(list(obj.keys())))
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)

def _extra_field(clean: str) -> str:
    obj = json.loads(clean)
    obj["note"] = "extracted successfully"
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)

def _chatty_tail(clean: str) -> str:
    return clean + "\n\nLet me know if you need anything else!"

CORRUPTIONS = [_fenced, _preamble, _trailing_comma, _single_quotes,
               _drop_field, _extra_field, _chatty_tail]


def make_rejected(clean_json: str, seed: int | None = None) -> str:
    if seed is not None:
        random.seed(seed)
    return random.choice(CORRUPTIONS)(clean_json)

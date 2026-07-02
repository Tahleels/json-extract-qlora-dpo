# src/config.py
from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any


@dataclass
class Config:
    raw: dict[str, Any]

    @property
    def seed(self) -> int: return self.raw["seed"]
    @property
    def base_model(self) -> str: return self.raw["model"]["base_id"]
    @property
    def field_keys(self) -> list[str]:
        return [f["key"] for f in self.raw["schema"]["fields"]]
    @property
    def system_prompt(self) -> str: return self.raw["system_prompt"].strip()
    @property
    def data(self) -> dict: return self.raw["data"]
    @property
    def dataset_repo(self) -> str: return self.raw["hub"]["dataset_repo"]


def load_config(path: str | Path = "configs/base.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))

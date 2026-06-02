import os
import yaml
from pathlib import Path

def load_system_instruction() -> str:
    prompt_file = Path(__file__).parent / "summary_prompt.yaml"

    with prompt_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data["meeting_analysis"]["system_instruction"]

def load_map_prompt() -> str:
    prompt_file = Path(__file__).parent / "map_prompt.yaml"

    with prompt_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data["map_prompt"]["template"]


def load_reduce_prompt() -> str:
    prompt_file = Path(__file__).parent / "reduce_prompt.yaml"

    with prompt_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data["reduce_prompt"]["template"]
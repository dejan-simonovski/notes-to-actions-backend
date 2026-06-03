import os
import yaml
from pathlib import Path

def load_system_instruction(context: str = "") -> str:
    perspective_instruction = (
        f"The intended reader is: {context.strip()}. "
        "Tailor the title, summary, and action items to highlight what is most relevant and actionable for their role."
        if context and context.strip()
        else "Provide a balanced output relevant to all stakeholders."
    )
    prompt_file = Path(__file__).parent / "summary_prompt.yaml"

    with prompt_file.open("r", encoding="utf-8") as f:
         data = yaml.safe_load(f)

    return data["meeting_analysis"]["system_instruction"].replace(
        "{perspective_instruction}", perspective_instruction
    )

def load_map_prompt() -> str:
    prompt_file = Path(__file__).parent / "map_prompt.yaml"

    with prompt_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data["map_prompt"]["template"]


def load_reduce_prompt(context: str = "") -> str:
    perspective_instruction = (
        f"The intended reader is: {context.strip()}. "
        "Tailor the summary to highlight decisions and action items most relevant to their role."
        if context and context.strip()
        else "Write the summary for a general audience covering all stakeholders."
    )

    return (
        "Write a coherent executive summary of this meeting based on the section summaries below.\n"
        "Preserve all decisions and action items with clear ownership where mentioned.\n"
        f"{perspective_instruction}\n"
        "{text}"
    )
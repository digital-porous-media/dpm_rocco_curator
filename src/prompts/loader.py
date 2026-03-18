"""Prompt loader utility for managing versioned YAML prompts."""

import yaml
from pathlib import Path
from jinja2 import Template


def load_prompt(name: str) -> dict:
    """Load a prompt from src/prompts/{name}.yaml.

    Args:
        name: Prompt name (e.g., 'evaluator', 'editor', 'content_screener')

    Returns:
        Dict with keys: version, description, system (optional), user

    Raises:
        FileNotFoundError: If prompt file does not exist
        yaml.YAMLError: If YAML parsing fails
    """
    prompt_dir = Path(__file__).parent
    prompt_file = prompt_dir / f"{name}.yaml"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, "r") as f:
        data = yaml.safe_load(f)

    return data


def render(template_str: str, **kwargs) -> str:
    """Render a Jinja2 template string with given variables.

    Args:
        template_str: Template string with {{ variable }} placeholders
        **kwargs: Variables to inject into template

    Returns:
        Rendered string
    """
    template = Template(template_str)
    return template.render(**kwargs)

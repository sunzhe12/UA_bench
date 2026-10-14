from pathlib import Path
import yaml


def load_config(path=None):
    """Load evaluation config from YAML file."""
    if path is None:
        path = Path(__file__).parent / "eval_config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



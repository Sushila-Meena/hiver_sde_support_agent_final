from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
def load_config():
    with open(ROOT/"configs/config.yaml",encoding="utf-8") as f:
        return yaml.safe_load(f)
def load_taxonomy():
    with open(ROOT/"configs/intent_taxonomy.yaml",encoding="utf-8") as f:
        return yaml.safe_load(f)["intents"]

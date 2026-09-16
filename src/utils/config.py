from __future__ import annotations
from pathlib import Path
import yaml


_DEFAULT_CONFIG_PATH_ = Path(__file__).resolve().parents[2]/"config"/"config.yaml"

def   load_config(path:str| Path = _DEFAULT_CONFIG_PATH_) -> Dict[str,Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"COnfig file not found: {Path}")
    with open(path,'r') as f:
        return yaml.safe_load(f)
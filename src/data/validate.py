from  __future__ import annotations
from dataclasses import dataclass,field
from typing import Dict,List

import pandas as pd

@dataclass
class ValidationResult:
    passed : bool
    errors:List[str] = field(default_factory = list)
    
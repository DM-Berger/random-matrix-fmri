from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

from pandas import DataFrame
from typing_extensions import Literal

Subgroups = Dict[str, List[Path]]
Pair = Tuple[Path, Path]
Observable = Literal[
    "eigs", "rigidity", "levelvar", "brody", "nnsd", "largest", "marchenko"
]
DataSummaryPaths = Dict[str, Dict[Observable, Path]]

PathFrame = DataFrame
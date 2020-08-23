from pathlib import Path
from typing import Dict, List, Tuple
from typing_extensions import Literal

Dataset = Dict[str, List[Path]]
Pair = Tuple[Path, Path]
Observable = Literal["eigs", "rigidity", "levelvar", "brody", "nnsd", "largest", "marchenko"]
DataSummaryPaths = Dict[str, Dict[Observable, Path]]

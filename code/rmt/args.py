import argparse
import numpy as np

from pathlib import Path
from typing import Dict, Any


def res(string: str) -> str:
    return str(Path(string).resolve())


def trim_parse(string: str) -> str:
    s = string.replace(" ", "")
    if s[0] != "(" or s[-1] != ")" or s.find(".") != -1:
        raise ValueError("`trim_indices` must by tuple of ints of length 2.")
    return str(s)


def smoother_parse(string: str) -> str:
    if string != "poly":
        raise ValueError("`smoother` argument currently only implemented for 'poly'.")
    return "poly"


def degree_parse(string: str) -> int:
    try:
        deg = int(string)
    except BaseException as e:
        raise ValueError("`degree` must be in int.") from e
    return deg


def bool_parse(string: str) -> bool:
    lower = string.lower()
    if lower == "true" or lower == "t":
        return True
    if lower == "false" or lower == "f":
        return False
    return bool(string)


def brody_parse(string: str) -> str:
    if string in ["mle", "spacing"]:
        return string
    raise ValueError("Brody fit method must be either 'mle' or 'spacing'.")


parser = argparse.ArgumentParser(description="Run with RMT args")
parser.add_argument(
    "-t", "--trim", metavar="<trim indices>", type=trim_parse, nargs=1, action="store", default="(1,-1)"
)
parser.add_argument(
    "-s", "--smoother", metavar="<smoother>", type=smoother_parse, nargs=1, action="store", default="poly"
)
parser.add_argument(
    "-d", "--degree", metavar="<smoother degree>", type=degree_parse, nargs=1, action="store", default=7
)
parser.add_argument("--detrend", metavar="<detrend>", type=bool_parse, nargs="?", action="store", default=False)
parser.add_argument("-b", "--brody", metavar="<brody>", type=brody_parse, nargs=1, action="store", default="mle")
parser.add_argument("--normalize", metavar="<normalize>", type=bool_parse, nargs="?", action="store", default=False)
parser.add_argument("--fullpre", action="store_true")

args = parser.parse_args()

# TRIM_IDX = "(1,-1)"  # must be indices of trims, tuple, no spaces
# UNFOLD_ARGS = {"smoother": "poly", "degree": 7, "detrend": False}
# LEVELVAR_ARGS = {"L": np.arange(0.5, 10, 0.1), "tol": 0.001, "max_L_iters": 50000}
# RIGIDITY_ARGS = {"L": np.arange(2, 20, 0.5)}
# BRODY_ARGS = {"method": "mle"}

trim = args.trim
smoother = args.smoother
degree = args.degree
brody = args.brody
normalize = args.normalize
is_fullpre = args.fullpre

TRIM_IDX = trim[0] if isinstance(trim, list) else trim
UNFOLD_ARGS = {
    "smoother": smoother[0] if isinstance(smoother, list) else smoother,
    "degree": degree[0] if isinstance(degree, list) else degree,
    "detrend": args.detrend,
}
LEVELVAR_ARGS = {"L": np.arange(0.5, 10, 0.1), "tol": 0.001, "max_L_iters": 50000}
RIGIDITY_ARGS = {"L": np.arange(2, 20, 0.5)}
BRODY_ARGS = {"method": brody[0] if isinstance(brody, list) else brody}
NORMALIZE = normalize[0] if isinstance(normalize, list) else normalize
FULLPRE = is_fullpre[0] if isinstance(is_fullpre, list) else is_fullpre


# yes, this is grotesque, but sometimes you need some damn singletons
# fmt: off
class Args:
    exists = False
    def __init__(self): # noqa
        if Args.exists: raise RuntimeError("Args object already exists.") # noqa

    @property
    def trim(self): return TRIM_IDX # noqa

    @trim.setter
    def trim(self, val: str):
        global TRIM_IDX
        TRIM_IDX = val

    @property
    def unfold(self): return UNFOLD_ARGS # noqa

    @unfold.setter
    def unfold(self, val: Dict[str, Any]):
        global UNFOLD_ARGS
        UNFOLD_ARGS = val

    @property
    def levelvar(self): return LEVELVAR_ARGS # noqa

    @levelvar.setter
    def levelvar(self, val: Dict[str, Any]):
        global LEVELVAR_ARGS
        LEVELVAR_ARGS = val

    @property
    def rigidity(self): return RIGIDITY_ARGS # noqa

    @rigidity.setter
    def rigidity(self, val: Dict[str, Any]):
        global RIGIDITY_ARGS
        RIGIDITY_ARGS = val

    @property
    def brody(self): return BRODY_ARGS # noqa

    @brody.setter
    def brody(self, val: Dict[str, Any]):
        global BRODY_ARGS
        BRODY_ARGS = val

    @property
    def normalize(self): return NORMALIZE # noqa

    @normalize.setter
    def normalize(self, val: bool):
        global NORMALIZE
        NORMALIZE = val

    @property
    def fullpre(self): return FULLPRE # noqa

    @fullpre.setter
    def fullpre(self, val: bool):
        global FULLPRE
        FULLPRE = val

    def print(self):
        from pprint import pprint
        print("=" * 80)
        print("Performing calculations for args:\n")
        print("TRIM_ARGS:      ", end="")
        pprint(self.trim)
        print("UNFOLD_ARGS:    ", end="")
        pprint(self.unfold)

        rig_min, rig_max = self.rigidity["L"].min(), np.round(self.rigidity["L"].max(), 2)
        rig_diff = np.round(np.diff(self.rigidity["L"])[0], 2)
        print(f"RIGIDITY_ARGS:  L=np.arange({rig_min}, {rig_max}, {rig_diff})")

        var_min, var_max = self.levelvar["L"].min(), np.round(self.levelvar["L"].max(), 2)
        var_diff = np.round(np.diff(self.levelvar["L"])[0], 2)
        tol, iters = self.levelvar["tol"], self.levelvar["max_L_iters"]
        print(f"LEVELVAR_ARGS:  L=np.arange({var_min}, {var_max}, {var_diff}), tol={tol}, max_L_iters={iters}")
        print(f"BRODY_ARGS:     {self.brody['method']}")
        print(f"NORMALIZE:      {self.normalize}")
        print(f"FULLPRE:        {self.fullpre}")
        print("")
        print("=" * 80)

    def cmd(self):
        """Print to stdout the commands to paste into run.sh."""
        smoother = self.unfold["smoother"]
        if isinstance(smoother, list):
            smoother = smoother[0]
        brody = self.brody["method"]
        if isinstance(brody, list):
            brody = brody[0]
        args = [
            f"--trim='{self.trim}'",
            f"--smoother='{smoother}'",
            f"--degree='{self.unfold['degree']}'",
            f"--brody='{brody}'",
        ]
        if self.unfold["detrend"]:
            args.append("--detrend")
        if self.normalize:
            args.append("--normalize")
        if self.fullpre:
            args.append("--fullpre")
        print(f"python3 run.py {' '.join(args)};")

ARGS = Args() # noqa

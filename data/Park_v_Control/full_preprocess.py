"""
The main challenge here is that we have to delete intermediates in order to
prevent completely filling up the HD with junk. However, if some intermediate
step fails, we want to keep the last relevant pre-processing step.
"""
import argparse
import nipype
import os
import pickle

from glob import glob
from nipype.interfaces.fsl import MCFLIRT, SliceTimer, BET
from pathlib import Path
from time import ctime
from typing import List, Tuple
from warnings import filterwarnings


DATA = Path(__file__).resolve().parent / "data"
LOGFILE = (
    Path(__file__).resolve().parent
    / f"{ctime().replace(' ', '_').replace(':', '.')}_pid{os.getpid()}__preprocess.log"
)

MCFLIRT_SUFFIX = "mcflirted.nii.gz"
SLICETIME_SUFFIX = "stationary.nii.gz"
BET_SUFFIX = "fullpre_stripped.nii.gz"


class Log:
    def __init__(self, logfile: Path = LOGFILE) -> None:
        self.log: List[str] = []
        if logfile.exists():
            os.remove(logfile)
        self.out = logfile

    def append(self, s: str) -> None:
        self.log.append(s)
        with open(self.out, "a+") as log:
            log.write(f"{s}\n")

    def print(self) -> None:
        print("\n".join(self.log))


def mcflirt(infile: Path) -> Path:
    # motion-correction FIRST supported by e.g.
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6736626/
    # given we are dealing with FSL
    outfile = Path(str(infile).replace("bold.nii.gz", MCFLIRT_SUFFIX))
    if outfile.exists():
        return outfile
    cmd = MCFLIRT(
        in_file=str(infile),
        out_file=str(outfile),
        output_type="NIFTI_GZ",
        save_mats=False,
        save_rms=False,
        stages=3,
        stats_imgs=False,
        terminal_output="stream",
        mean_vol=False,  # speed up things
    )

    results = cmd.run()
    return Path(results.outputs.out_file)


def slicetime_correct(infile: Path, timings: Path, TR: float, slice_direction: str = "z") -> Path:
    """There are considerable inconsistencies in
    """
    outfile = Path(str(infile).replace(MCFLIRT_SUFFIX, SLICETIME_SUFFIX))
    cmd = SliceTimer()
    cmd.inputs.in_file = str(infile)
    cmd.inputs.custom_timings = str(timings)
    cmd.inputs.time_repetition = TR
    cmd.inputs.out_file = str(infile).replace(MCFLIRT_SUFFIX, SLICETIME_SUFFIX)
    cmd.inputs.output_type = "NIFTI_GZ"
    cmd.inputs.slice_direction = slice_direction
    if outfile.exists():
        return outfile
    cmd.run()
    return outfile  # different interface than above


def skullstrip(infile: Path) -> Tuple[Path, Path]:
    cmd = BET()
    cmd.inputs.in_file = str(infile)
    cmd.inputs.out_file = str(infile).replace(SLICETIME_SUFFIX, BET_SUFFIX)
    cmd.inputs.output_type = "NIFTI_GZ"
    cmd.inputs.functional = True
    cmd.inputs.mask = True
    results = cmd.run()
    return results.outputs.out_file, results.outputs.mask_file


def fileparse(string: str) -> Path:
    try:
        p = Path(string).resolve()
    except BaseException as e:
        raise ValueError(f"Could not convert string {string} to valid pathlib Path.") from e
    return p


def direction_parse(string: str) -> int:
    if string in ["i", "x", "1"]:
        return 1
    elif string in ["j", "y", "2"]:
        return 2
    elif string in ["k", "z", "3"]:
        return 3
    else:
        raise ValueError(
            "Argument to --direction must be one of ['i', 'j', 'k', '1', '2', '3', 'x', 'y', 'z']."
        )


parser = argparse.ArgumentParser(description="Run with RMT args")
parser.add_argument(
    "-i", "--infile", metavar="<input bold.nii.gz>", type=fileparse, nargs=1, action="store"
)
# parser.add_argument(
#     "-o", "--outfile", metavar="<output bold.nii.gz>", type=fileparse, nargs=1, action="store"
# )
parser.add_argument(
    "-t", "--timings", metavar="<slicetimes.txt>", type=fileparse, nargs=1, action="store"
)
parser.add_argument("--TR", metavar="<Time of Repetition>", type=float, nargs=1, action="store")
parser.add_argument(
    "-d", "--direction", metavar="<Slice Direction>", type=direction_parse, nargs=1, action="store"
)
parser.add_argument("--flag", metavar="<flag>", nargs="?", action="store", default=False)

raw_args = vars(parser.parse_args())

# TRIM_IDX = "(1,-1)"  # must be indices of trims, tuple, no spaces
# UNFOLD_ARGS = {"smoother": "poly", "degree": 7, "detrend": False}
# LEVELVAR_ARGS = {"L": np.arange(0.5, 10, 0.1), "tol": 0.001, "max_L_iters": 50000}
# RIGIDITY_ARGS = {"L": np.arange(2, 20, 0.5)}
# BRODY_ARGS = {"method": "mle"}

args = {}
for key, val in raw_args.items():
    args[key] = val[0] if isinstance(val, list) else val  # deal with strange wrapping behaviour

infile = args["infile"]
# outfile = args["outfile"]
timings = args["timings"]
direction = args["direction"]
TR = args["TR"]

log = Log(LOGFILE)
bet_out = Path(str(infile).replace("bold.nii.gz", BET_SUFFIX)).resolve()
if bet_out.exists():
    log.append(f"File {infile.relative_to(DATA)} already preprocessed!")
    log.print()
else:
    try:
        mcflirt_out = mcflirt(infile)
        log.append(f"MCFLIRTed image saved to .......... {mcflirt_out}")
        slicetime_out = slicetime_correct(mcflirt_out, timings, TR=TR, slice_direction=direction)
        log.append(f"SliceTime-corrected image saved to: {slicetime_out}")
        os.unlink(mcflirt_out)
        log.append(f"Removed intermediate MCFLIRT image: {mcflirt_out}")
        stripped, mask = skullstrip(slicetime_out)
        log.append(f"Saved skull-stripped image to ..... {stripped}")
        log.append(f"Saved brain mask image to ......... {mask}")
        os.unlink(slicetime_out)
        log.append(f"Removed intermediate STC image .... {slicetime_out}")
        log.print()
        os.unlink(log.out)
    except BaseException as e:
        log.append("^^^ Preprocessing failed after the step listed above. ^^^")
        log.print()
        raise e

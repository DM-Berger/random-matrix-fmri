import numpy as np

from glob import glob
from pathlib import Path


def print_extrema() -> None:
    DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

    np.set_printoptions(precision=6, linewidth=150)

    # shape_paths = np.sort([Path(file) for file in glob(f"{DATA_ROOT}/**/*shapes*.npy", recursive=True)])
    shapes = [
        np.load(file, allow_pickle=True)
        for file in glob(f"{DATA_ROOT}/**/*shapes*.npy", recursive=True)
    ]
    paths = np.sort(
        [Path(file).resolve() for file in glob(f"{DATA_ROOT}/**/*eigs*.npy", recursive=True)]
    )
    eigs_all = [
        np.load(path, allow_pickle=True)
        for path in glob(f"{DATA_ROOT}/**/*eigs*.npy", recursive=True)
    ]
    # parents = [path.parent.relative_to(DATA_ROOT) for path in paths]
    parents = [path.parent for path in paths]
    mins, maxs = (
        np.array([eigs[:20] for eigs in eigs_all]),
        np.array([eigs[-20:] for eigs in eigs_all]),
    )
    for eigs, shape, mn, mx, parent in zip(eigs_all, shapes, mins, maxs, parents):
        print(str(parent))
        print(f"shape: {shape}")
        # print(f"mins:  {np.round(mn, 1)}")
        print(f"maxs:  {np.round(mx, 1)}")
        print(f"mean:  {np.round(np.mean(eigs), 1)}")
        print("=" * 80)


if __name__ == "__main__":
    print_extrema()

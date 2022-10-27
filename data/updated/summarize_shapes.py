from pathlib import Path

import pandas as pd
from ants import ANTsImage, image_read
from pandas import DataFrame
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

DATA = Path(__file__).resolve().parent
DATASETS = sorted(filter(lambda p: p.is_dir(), Path(".").glob("*")))


def get_fmri_shapes(fmri: Path) -> DataFrame:
    img: ANTsImage = image_read(fmri)
    x, y, z, t = img.shape
    xx, yy, zz, TR = img.spacing
    return DataFrame(
        {
            "data": data.name,
            "x_n": x,
            "y_n": y,
            "z_n": z,
            "t": t,
            "x_mm": xx,
            "y_mm": yy,
            "z_mm": zz,
            "TR": TR,
        },
        index=[0],
    )


if __name__ == "__main__":
    dfs = []
    for data in DATASETS:
        fmris = sorted(data.rglob("*bold.nii.gz"))
        print(f"Collecting shape info for {data.name}")
        dfs.extend(process_map(get_fmri_shapes, fmris))

    df = pd.concat(dfs, axis=0, ignore_index=True)
    out = DATA / "shape_summary.csv"
    df.to_csv(out)
    print(f"Saved shape summary to {out}")

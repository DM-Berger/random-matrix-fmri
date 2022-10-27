from pathlib import Path

import pandas as pd
from ants import ANTsImage, image_read
from pandas import DataFrame
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

DATA = Path(__file__).resolve().parent
DATASETS = sorted(filter(lambda p: p.is_dir(), Path(".").glob("*")))


def get_fmri_shapes(fmri: Path) -> DataFrame:
    img: ANTsImage = image_read(str(fmri))
    x, y, z, t = img.shape
    xx, yy, zz, TR = img.spacing
    orient = img.get_orientation()
    return DataFrame(
        {
            "data": data.name,
            "x_n": x,
            "y_n": y,
            "z_n": z,
            "x_mm": xx,
            "y_mm": yy,
            "z_mm": zz,
            "t": t,
            "TR": TR,
            "orient": orient,
        },
        index=[str(fmri)],
    )


if __name__ == "__main__":
    dfs = []
    for data in DATASETS:
        fmris = sorted(data.rglob("*bold.nii.gz"))
        dfs.extend(
            process_map(
                get_fmri_shapes,
                fmris,
                chunksize=1,
                desc=f"Collecting shape info for {data.name}",
            )
        )

    df = pd.concat(dfs, axis=0, ignore_index=True)
    out = DATA / "shape_summary.csv"
    df.to_csv(out)
    print(f"Saved shape summary to {out}")
    df = df.round(2).drop_duplicates().sort_values(by=["data", "x_n", "x_mm", "t", "TR"])
    print(df)
    """
    Result:

    data                          x_n     y_n    z_n    t    x_mm    y_mm    z_mm     TR
    ---------------------------  -----  -----  -----  ---  ------  ------  ------  -------
    Park_v_Control                  80     80     43  149    3       3       3        2.4
    Park_v_Control                  80     80     43  300    3       3       3        2.4
    Park_v_Control                  96    114     96  149    2       2       2        2.4
    Rest_v_LearningRecall           64     64     36  195    3       3       3        2
    Rest_w_Bilinguiality           100    100     72  823    1.8     1.8     1.8      0.88
    Rest_w_Bilinguiality           100     96     72  823    1.8     1.8     1.8      0.88
    Rest_w_Bilinguiality           100    100     72  823    1.8     1.8     1.8      0.93
    Rest_w_Depression_v_Control    112    112     25  100    1.96    1.96    5        2.5
    Rest_w_Healthy_v_OsteoPain      64     64     36  244    3.44    3.44    3        2.5
    Rest_w_Healthy_v_OsteoPain      64     64     36  292    3.44    3.44    3        2.5
    Rest_w_Healthy_v_OsteoPain      64     64     36  300    3.44    3.44    3        2.5
    Rest_w_Older_v_Younger          74     74     32  300    2.97    2.97    4        2
    Rest_w_VigilanceAttention       64     64     35  300    3       3       3     3000
    Rest_w_VigilanceAttention      128    128     70  300    1.5     1.5     1.5   3000
    Rest_w_VigilanceAttention      200     60     40  150    0.75    0.75    0.75  4000

    """

import numpy as np

from rmt.constants import DATASETS_FULLPRE

for dataset_name, dataset in DATASETS_FULLPRE.items():
    print(f"{'='*80}\n{dataset_name.upper()}")
    for subgroup, eigpaths in dataset.items():
        shape_paths = [str(p.resolve()).replace("eigs-", "shapes-") for p in eigpaths]
        shapes = [np.array(np.load(shape), dtype=int) for shape in shape_paths]
        ns, ts = np.array([shape[0] for shape in shapes]), np.array([shape[1] for shape in shapes])
        n_mean, t_mean = int(np.round(np.mean(ns))), int(np.round(np.mean(ts)))
        print(f"{'-'*80}\n{subgroup}")
        print(f"N subjects: {len(eigpaths)}")
        print("        {:>7}   {:>7}".format("n", "t"))
        print("   min: {:7d}   {:7d}".format(int(ns.min()), int(ts.min())))
        print("   max: {:7d}   {:7d}".format(int(ns.max()), int(ts.max())))
        print("  mean: {:7d}   {:7d}".format(n_mean, t_mean))

import importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("m", r"model_new\03_XGB训练_二分类.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

rate = pd.read_csv(r"model_new\data\rate.csv")
r = rate["rate"].values
print("threshold | n_points")
for t in [0.15, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
    pts = m._detect_cusum_changepoints(r, threshold=t)
    print(f"{t:.2f}      | {len(pts)}")
    if t in [1.0, 2.0]:
        print(f"         indices: {pts[:10]}...")

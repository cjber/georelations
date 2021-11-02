import pandas as pd
from pathlib import Path

paths = list(Path("csv_logs/").glob("*/*/metrics.csv"))
dfs = (
    pd.concat(
        [
            pd.read_csv(path).assign(seed=path._parts[1], model=path._parts[2])
            for path in paths
        ]
    )
    .fillna(method="bfill")
    .fillna(method="ffill")
)

dfs.to_csv("paper/pdata/metrics.csv", index=False)

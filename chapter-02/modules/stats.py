import numpy as np
import pandas as pd
from scipy import stats
from collections import namedtuple


CIResult = namedtuple("CIResult", ["mean", "ci_low", "ci_high"])
def mean_ci(values, confidence=0.95):
    m = values.mean(); n = len(values); se = values.std() / np.sqrt(n)
    z = stats.norm.ppf((1 + confidence) / 2)
    return CIResult(mean=m, ci_low=m - z*se, ci_high=m + z*se)

def summary_stats(df, columns):
    rows = []
    for col in columns:
        r = mean_ci(df[col])
        rows.append({"variable": col, "mean": round(r.mean,4),
                     "ci_low": round(r.ci_low,4), "ci_high": round(r.ci_high,4)})
    return pd.DataFrame(rows)
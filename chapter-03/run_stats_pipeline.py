# run_stats_pipeline.py
import pandas as pd
from modules.stats import mean_ci, summary_stats

url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"

def main():
    df = pd.read_csv(url, index_col=0)

    # Single column
    result = mean_ci(df["TV"])
    print(f"TV — Mean: {result.mean:.2f}, 95% CI: ({result.ci_low:.2f}, {result.ci_high:.2f})")

    # All advertising channels
    summary = summary_stats(df, columns=["TV", "radio", "newspaper"])
    print(summary)
    
if __name__ == "__main__":
    main()
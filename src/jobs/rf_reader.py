import pandas as pd

rf = pd.read_csv("metrics_summary_rf.csv")

# add the missing column
rf["method"] = "RF"

# put it in a convenient position (after regime, before n_seeds)
cols = ["dataset", "regime", "method", "n_seeds",
        "AP_mean", "AP_std", "AUPR_mean", "AUPR_std",
        "ROC_AUC_mean", "ROC_AUC_std"]
rf = rf[cols]          # re‑order

rf.to_csv("rf_with_method.csv", index=False)
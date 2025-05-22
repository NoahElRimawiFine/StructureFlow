#!/usr/bin/env python
# combine_metric_tables.py ----------------------------------------------------
"""
Usage
-----
python combine_metric_tables.py \
       --baseline  baselines.csv \
       --sf2m      sf2m.csv \
       --rf        rf.csv \
       --out       all_metrics.csv
"""

import argparse
import pandas as pd

def load_baseline(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df = df.rename(columns={
        "mean_AP":  "AP_mean",
        "std_AP":   "AP_std",
        "mean_AUC": "ROC_AUC_mean",
        "std_AUC":  "ROC_AUC_std",
    })
    df["n_seeds"]     = ""          # unknown
    df["AUPR_mean"]   = ""
    df["AUPR_std"]    = ""
    # ensure column order
    cols = ["dataset","regime","method","n_seeds",
            "AP_mean","AP_std","AUPR_mean","AUPR_std",
            "ROC_AUC_mean","ROC_AUC_std"]
    return df[cols]

def load_sf2m(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df["method"] = "StructureFlow"
    cols = ["dataset","regime","method","n_seeds",
            "AP_mean","AP_std","AUPR_mean","AUPR_std",
            "ROC_AUC_mean","ROC_AUC_std"]
    return df[cols]

def load_rf(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df["method"] = "RF"
    # rf csv has AP_mean/AP_std at the end; ensure same order
    cols = ["dataset","regime","method","n_seeds",
            "AP_mean","AP_std","AUPR_mean","AUPR_std",
            "ROC_AUC_mean","ROC_AUC_std"]
    return df[cols]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--sf2m",     required=True)
    p.add_argument("--rf",       required=True)
    p.add_argument("--out",      default="all_metrics.csv")
    args = p.parse_args()

    df_base = load_baseline(args.baseline)
    df_sf2m = load_sf2m(args.sf2m)
    df_rf   = load_rf(args.rf)

    all_df = pd.concat([df_base, df_sf2m, df_rf], ignore_index=True)
    all_df.to_csv(args.out, index=False)
    print(f"\nCombined table saved → {args.out}\n")
    print(all_df.to_string(index=False))

if __name__ == "__main__":
    main()

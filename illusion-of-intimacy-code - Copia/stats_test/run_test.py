#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon

# List of emotion columns
emotion_cols = [
    'anger', 'disgust', 'fear', 'sadness',
    'anticipation', 'surprise', 'optimism',
    'joy', 'love', 'trust', 'pessimism'
]

def main(csv_path, threshold=0.05):
    # 1) Load data
    df = pd.read_csv(csv_path)

    # 2) Mask out any turn where emotion < threshold
    df[emotion_cols] = df[emotion_cols].where(df[emotion_cols] >= threshold, np.nan)

    # 3) Compute per-dialogue, per-party averages (ignoring NaNs)
    agg = (
        df
        .groupby(['id', 'party'])[emotion_cols]
        .mean()
        .reset_index()
    )

    # 4) Split USER vs Chatbot, drop 'party', set ID as index
    user = agg[agg.party == 'USER']\
             .drop(columns='party')\
             .set_index('id')\
             .rename(columns={e: f"{e}_user" for e in emotion_cols})
    bot = agg[agg.party == 'Chatbot']\
             .drop(columns='party')\
             .set_index('id')\
             .rename(columns={e: f"{e}_chatbot" for e in emotion_cols})

    # 5) Join on ID
    merged = user.join(bot, how='inner')

    # 6) Open output file
    with open("emotional_mirroring_results.txt", "w") as f:
        f.write(f"Masking threshold: {threshold}\n")
        f.write("Interpretation: comparing USER vs. Chatbot average emotion per dialogue\n\n")

        for emo in emotion_cols:
            u = merged[f"{emo}_user"]
            b = merged[f"{emo}_chatbot"]

            # keep only dialogues with valid averages on both sides
            valid = u.notna() & b.notna()
            u_valid = u[valid]
            b_valid = b[valid]
            diff = b_valid - u_valid
            n = len(diff)

            f.write(f"=== {emo.upper()} (n={n} dialogues) ===\n")
            if n < 2:
                f.write("  Not enough data after thresholding. Skipping.\n\n")
                continue

            # decide whether to assume normality for large n
            if n > 5000:
                normal = True
            elif n >= 3:
                _, p_sw = shapiro(diff)
                normal = (p_sw > 0.05)
            else:
                normal = False

            # choose paired test
            if normal:
                stat, p_val = ttest_rel(b_valid, u_valid, nan_policy='omit')
                test_name = "Paired t-test"
            else:
                stat, p_val = wilcoxon(b_valid, u_valid)
                test_name = "Wilcoxon signed-rank"

            # interpret
            if p_val >= 0.05:
                f.write(f"  {test_name}: p = {p_val:.4f} → no significant difference\n\n")
            else:
                mean_u = u_valid.mean()
                mean_b = b_valid.mean()
                direction = "higher" if mean_b > mean_u else "lower"
                f.write(f"  {test_name}: p = {p_val:.4f} → Chatbot is significantly {direction} than User\n\n")

    print("Done. See 'emotional_mirroring_results.txt' for interpretations.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_csv>")
        sys.exit(1)
    main(sys.argv[1])

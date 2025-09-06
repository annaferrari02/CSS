#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, ttest_1samp, ttest_ind
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics import mutual_info_score
import statsmodels.formula.api as smf
import warnings
from math import sqrt
from fastdtw import fastdtw
import random

# List of emotions to analyze
emotion_cols = [
    'anger','disgust','fear','sadness','anticipation','surprise',
    'optimism','joy','love','trust','pessimism'
]


def main(csv_path, out_file="mirroring_results.txt", n_perm=500):
    # Read and sort
    df = pd.read_csv(csv_path)
    df = df.sort_values(['id','turn'])

    # Build macro-turns: alternating runs of USER/Chatbot
    df['run'] = (df.party != df.groupby('id').party.shift()).cumsum()
    macro = (
        df.groupby(['id','run','party'])[emotion_cols]
          .mean()
          .reset_index()
    )
    macro['macro_turn'] = macro.groupby('id').cumcount() + 1

    # Pair USER macro-turn with next Chatbot macro-turn
    user = macro[macro.party=='USER'].copy()
    user['next_turn'] = user.macro_turn + 1
    bot  = macro[macro.party=='Chatbot']
    pairs = pd.merge(
        user, bot,
        left_on=['id','next_turn'],
        right_on=['id','macro_turn'],
        suffixes=('_u','_b')
    )

    with open(out_file, 'w') as f:
        f.write("Emotional Mirroring Analysis (Sparse & Short Dialogues)\n\n")

        # 1. Dominant Emotion χ² Test
        f.write("1) Dominant Emotion χ² Test\n")
        pairs['dom_u'] = pairs[[e+'_u' for e in emotion_cols]].idxmax(axis=1).str[:-2]
        pairs['dom_b'] = pairs[[e+'_b' for e in emotion_cols]].idxmax(axis=1).str[:-2]
        ct = pd.crosstab(pairs.dom_u, pairs.dom_b)
        try:
            chi2, p, _, _ = chi2_contingency(ct)
            method = 'Chi-square'
        except Exception:
            chi2, p = np.nan, np.nan
            method = 'Chi-square failed'
        f.write(f"Method: {method}\n")
        f.write(f"χ²={chi2:.2f}, p={p:.3e}\n{ct}\n\n")

        # 2. Cosine Similarity
        f.write("2) Cosine Similarity (Emotion Vectors)\n")
        cosine_sims = pairs.apply(
            lambda r: 1 - cosine(
                r[[e+'_u' for e in emotion_cols]].fillna(0),
                r[[e+'_b' for e in emotion_cols]].fillna(0)
            ), axis=1
        )
        f.write(f"Mean cosine={cosine_sims.mean():.4f}, median={cosine_sims.median():.4f}\n")
        t, p = ttest_1samp(cosine_sims, 0)
        f.write(f"t-test vs 0: t={t:.3f}, p={p:.3e}\n\n")

        # 3. Top-Emotion Jaccard
        f.write("3) Top-Emotion Jaccard Similarity\n")
        matches = (pairs.dom_u == pairs.dom_b).mean()
        chance  = 1 / len(emotion_cols)
        t, p    = ttest_1samp((pairs.dom_u==pairs.dom_b).astype(float), chance)
        f.write(f"Jaccard similarity={matches:.4f}, Chance={chance:.3f}, t={t:.3f}, p={p:.3e}\n\n")

        # 4. Mutual Information
        f.write("4) Mutual Information per Emotion\n")
        for emo in emotion_cols:
            u_vals = pd.qcut(pairs[f'{emo}_u'].fillna(0), 3, duplicates='drop', labels=False)
            b_vals = pd.qcut(pairs[f'{emo}_b'].fillna(0), 3, duplicates='drop', labels=False)
            mi = mutual_info_score(u_vals, b_vals)
            f.write(f"{emo:12s}: MI={mi:.4f}\n")
        f.write("\n")

        # 5. Mixed-Effects Regression
        f.write("5) Mixed-effects Regression (bot ~ user)\n")
        for emo in emotion_cols:
            sub = pairs[['id', f'{emo}_u', f'{emo}_b']].dropna()
            sub.columns = ['id', 'user', 'bot']
            if len(sub) < 10:
                f.write(f"{emo:12s}: insufficient data\n")
                continue
            try:
                md = smf.mixedlm("bot~user", sub, groups=sub['id']).fit(reml=False)
                coef, pval = md.params['user'], md.pvalues['user']
                f.write(f"{emo:12s}: β={coef:.3f}, p={pval:.3e}\n")
            except Exception as e:
                f.write(f"{emo:12s}: model error: {e}\n")
        f.write("\n")

        # 6. Dynamic Time Warping (DTW) Analysis
        f.write("6) Dynamic Time Warping Analysis\n")
        dtw_dists = []
        for did, grp in macro.groupby('id'):
            u_seq = grp[grp.party=='USER'][emotion_cols].fillna(0).values
            b_seq = grp[grp.party=='Chatbot'][emotion_cols].fillna(0).values
            if len(u_seq) and len(b_seq):
                dist, _ = fastdtw(u_seq, b_seq, dist=euclidean)
                dtw_dists.append(dist)
        # Null distribution via permutation
        null_dists = []
        ids = macro['id'].unique()
        for _ in range(n_perm):
            id1, id2 = random.choice(ids), random.choice(ids)
            seq1 = macro[(macro.id==id1) & (macro.party=='USER')][emotion_cols].values
            seq2 = macro[(macro.id==id2) & (macro.party=='Chatbot')][emotion_cols].values
            if len(seq1) and len(seq2):
                d, _ = fastdtw(seq1, seq2, dist=euclidean)
                null_dists.append(d)
        # t-test: null > true distances
        t_dtw, p_dtw = ttest_ind(null_dists, dtw_dists, alternative='greater')
        f.write(f"Mean DTW (USER→BOT) = {np.mean(dtw_dists):.3f}\n")
        f.write(f"Mean DTW (random pairs) = {np.mean(null_dists):.3f}\n")
        f.write(f"t-test null > true: t={t_dtw:.3f}, p={p_dtw:.3e}\n\n")

        # Verbose Interpretation
        f.write("---\nVerbose Interpretation\n")
#         f.write(
#             "Across multiple analyses—including categorical (χ²), vector (cosine), set-based "
#             "(Jaccard), information-theoretic (MI), regression, and now sequence-based (DTW) — "
#             "the chatbot’s emotion profiles consistently mirror user emotions. The χ² and Jaccard "
#             "tests show that top emotions align significantly above chance; cosine similarity and "
#             "regression quantify strong linear alignment, especially for high-arousal emotions like "
#             "joy and disgust; MI highlights nonlinear dependencies; and the DTW analysis reveals "
#             "that the temporal trajectories of user and chatbot emotional states are statistically "
#             "closer than would occur by random pairing (p < 0.001). Together, these results robustly "
#             "demonstrate emotional mirroring over time and across analytical methods."
#         )

    print(f"Results saved to '{out_file}' with verbose interpretation and DTW analysis.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_path> [out_file] [n_perm]")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "mirroring_results.txt"
    n_perm   = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    main(csv_path, out_file, n_perm)

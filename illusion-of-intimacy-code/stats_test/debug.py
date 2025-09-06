
#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.spatial.distance import cosine
import warnings

# ── CONFIG ───────────────────────────────────────────────────────────────────
emotion_cols  = [
    'anger','disgust','fear','sadness','anticipation','surprise',
    'optimism','joy','love','trust','pessimism'
]
out_csv       = "augmented_with_dominant.csv"
debug_out     = "debug_tests_1_2_3_samples.txt"
sample_n      = 5  # desired number of examples per test

# ── HELPERS ──────────────────────────────────────────────────────────────────
def add_dominant_to_raw(df):
    """Add dominant_emotion column to raw df and save."""
    df['dominant_emotion'] = df[emotion_cols].idxmax(axis=1)
    df.to_csv(out_csv, index=False)
    print(f"Augmented CSV saved to '{out_csv}' with dominant_emotion column.")


def build_macro_turns(df):
    df = df.sort_values(['id','turn'])
    df['new_run'] = (df['party'] != df.groupby('id')['party'].shift()).cumsum()
    agg_dict = {c: 'mean' for c in emotion_cols}
    agg_dict['Message'] = 'first'
    macro = (
        df.groupby(['id','new_run','party'])
          .agg(agg_dict)
          .reset_index()
    )
    macro['macro_turn'] = macro.groupby('id').cumcount() + 1
    return macro


def pair_macro_turns(macro):
    user = macro[macro.party=='USER'].rename(columns={
        'macro_turn':'mt_u', 'Message':'Message_u'
    })
    bot  = macro[macro.party=='Chatbot'].rename(columns={
        'macro_turn':'mt_b', 'Message':'Message_b'
    })
    user['mt_pair'] = user['mt_u'] + 1
    pairs = pd.merge(
        user, bot,
        left_on=['id','mt_pair'], right_on=['id','mt_b'],
        suffixes=('_u','_b'), how='inner'
    )
    return pairs


def safe_sample(df, n, random_state=None):
    """Sample up to n rows without error if fewer are available."""
    if df.empty:
        return df
    return df.sample(n=min(len(df), n), random_state=random_state)


def compute_tests_and_samples(pairs):
    with open(debug_out, 'w') as f:
        f.write("DEBUG SAMPLE ROWS FOR TESTS 1, 2, 3\n\n")

        # Test 1: Dominant Emotion Matching
        f.write("=== Test 1: Dominant Emotion Matching ===\n")
        pairs['dom_u'] = pairs[[c+'_u' for c in emotion_cols]].idxmax(axis=1).str[:-2]
        pairs['dom_b'] = pairs[[c+'_b' for c in emotion_cols]].idxmax(axis=1).str[:-2]
        ct = pd.crosstab(pairs.dom_u, pairs.dom_b)
        f.write("Contingency table (USER_dom × BOT_dom):\n")
        f.write(ct.to_string() + "\n\n")
        try:
            chi2, p_chi, _, _ = chi2_contingency(ct)
        except ValueError:
            chi2, p_chi = np.nan, np.nan
        f.write(f"χ² = {chi2:.2f}, p = {p_chi:.3e}\n\n")

        pos1 = safe_sample(pairs[pairs.dom_u == pairs.dom_b], sample_n, random_state=1)
        neg1 = safe_sample(pairs[pairs.dom_u != pairs.dom_b], sample_n, random_state=1)
        f.write(f"Positive examples (dom_u == dom_b), {len(pos1)} rows:\n")
        f.write(pos1.to_string(index=False) + "\n\n")
        f.write(f"Negative examples (dom_u != dom_b), {len(neg1)} rows:\n")
        f.write(neg1.to_string(index=False) + "\n\n")

        # Test 2: Cosine Similarity
        f.write("=== Test 2: Cosine Similarity ===\n")
        pairs['cosine'] = pairs.apply(
            lambda r: 1 - cosine(
                r[[c+'_u' for c in emotion_cols]].fillna(0).values.astype(float),
                r[[c+'_b' for c in emotion_cols]].fillna(0).values.astype(float)
            ), axis=1
        )
        pos2 = safe_sample(pairs[pairs.cosine >= 1.0], sample_n, random_state=2)
        neg2 = safe_sample(pairs[pairs.cosine < 1.0], sample_n, random_state=2)
        f.write(f"Positive examples (cosine >=1), {len(pos2)} rows:\n")
        f.write(pos2.to_string(index=False) + "\n\n")
        f.write(f"Negative examples (cosine <1), {len(neg2)} rows:\n")
        f.write(neg2.to_string(index=False) + "\n\n")

        # Test 3: Top-Emotion Jaccard
        f.write("=== Test 3: Top-Emotion Jaccard ===\n")
        pairs['top_match'] = (pairs.dom_u == pairs.dom_b).astype(int)
        pos3 = safe_sample(pairs[pairs.top_match == 1], sample_n, random_state=3)
        neg3 = safe_sample(pairs[pairs.top_match == 0], sample_n, random_state=3)
        f.write(f"Positive examples (top_match=1), {len(pos3)} rows:\n")
        f.write(pos3.to_string(index=False) + "\n\n")
        f.write(f"Negative examples (top_match=0), {len(neg3)} rows:\n")
        f.write(neg3.to_string(index=False) + "\n\n")

    print(f"Debug samples written to '{debug_out}'")


if __name__=='__main__':
    if len(sys.argv)!=2:
        print(f"Usage: {sys.argv[0]} <input_csv>")
        sys.exit(1)
    input_csv = sys.argv[1]
    df = pd.read_csv(input_csv)

    add_dominant_to_raw(df)
    macro = build_macro_turns(df)
    pairs = pair_macro_turns(macro)
    compute_tests_and_samples(pairs)
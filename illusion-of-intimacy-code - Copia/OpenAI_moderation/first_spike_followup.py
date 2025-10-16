#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
import multiprocessing as mp

# 1. Configuration
csv_path     = "/home/mhchu/AI-Companion/human-ai/data/data/combined_data.csv"
output_txt   = "user_spike_population_baseline_ttests.txt"
threshold    = 0.5
categories   = ["sexual", "violence", "harassment"]

# shared DataFrame, set in each worker
df = None

def init_worker(df_):
    """Initializer for each Pool worker, to set the global df."""
    global df
    df = df_

def process_category(cat):
    """
    For a single category:
      - compute population mean of Chatbot's scores for that category
      - find each dialogue's first USER spike (score > threshold)
      - collect the Chatbot's immediate next-turn score
      - run a one-sample t-test of post-spike scores vs. population mean
    Returns a dict of results.
    """
    # 1) population mean over all Chatbot turns
    bot_all = df[df.party == 'Chatbot'][cat].dropna()
    pop_mean = bot_all.mean()

    # 2) collect post-spike Chatbot scores
    post_scores = []
    for dlg_id, grp in df.groupby('id'):
        # find first USER spike
        user_spikes = grp[(grp.party=='USER') & (grp[cat] > threshold)]
        if user_spikes.empty:
            continue
        turn0 = user_spikes.turn.min()
        # immediate bot response
        bot_row = grp[(grp.party=='Chatbot') & (grp.turn == turn0 + 1)]
        if bot_row.empty:
            continue
        post_scores.append(bot_row[cat].iloc[0])

    post_scores = np.array(post_scores, dtype=float)
    n = len(post_scores)

    # 3) one-sample t-test vs pop_mean
    if n > 0:
        t_stat, p_val = ttest_1samp(post_scores, pop_mean, nan_policy='omit')
        mean_post = post_scores.mean()
        std_post  = post_scores.std(ddof=1)
    else:
        t_stat = np.nan
        p_val   = np.nan
        mean_post = np.nan
        std_post  = np.nan

    return {
        'category':   cat,
        'pop_mean':   pop_mean,
        'n':          n,
        'mean_post':  mean_post,
        'std_post':   std_post,
        't_stat':     t_stat,
        'p_val':      p_val
    }

if __name__ == '__main__':
    # 2. Load data once
    df_master = pd.read_csv(csv_path)

    # 3. Parallel processing of categories
    with mp.Pool(initializer=init_worker, initargs=(df_master,)) as pool:
        results = pool.map(process_category, categories)

    # 4. Write results to text file
    with open(output_txt, 'w') as f:
        f.write("One-Sample t-Test of Chatbot Post-Spike Scores vs. Population Mean\n")
        f.write("Baseline = population mean of chatbot scores for each category\n")
        f.write("="*70 + "\n\n")
        for res in results:
            f.write(f"Category: {res['category']}\n")
            f.write(f"  Population mean       = {res['pop_mean']:.4f}\n")
            f.write(f"  N post-spike scores   = {res['n']}\n")
            f.write(f"  Mean post-spike score = {res['mean_post']:.4f}\n")
            f.write(f"  Std  post-spike score = {res['std_post']:.4f}\n")
            f.write(f"  t-statistic           = {res['t_stat']:.3f}\n")
            f.write(f"  p-value               = {res['p_val']:.3e}\n")
            f.write("\n")

    print(f"✅ Results written to {output_txt}")

#!/usr/bin/env python3
"""
plot_score_density.py

A script to read all CSV files in a specified folder (same grouping logic as before),
and plot overlaid density curves of the 'score' column (0–1) for each group/bar.

Special cases:
  - human_submissions_emo.csv → two densities: romantic vs non‑romantic subreddits
  - human_ai_chatlogs_emo.csv → two densities: USER vs Chatbot
  - all others → one density per file

Exports both PDF and PNG.
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# relationship groups
romantic_relationships = [
    "relationships", "relationship_advice", "dating_advice", "LongDistance",
    "Marriage", "BreakUps", "datingoverthirty", "survivinginfidelity",
    "polyamory", "r4r", "OkCupid", "Tinder", "Bumble", "hingeapp",
    "dating", "exnocontact", "divorce", "relationship_advice_for_men",
    "AnxiousAttachment"
]
non_romantic_relationships = [
    "FriendshipAdvice", "family", "Parenting", "raisedbynarcissists",
    "JustNoFamily", "JustNoSO", "JustNoTalk", "entitledparents",
    "MomForAMinute", "friendship"
]

# x‑tick label mapping (same keys as before)
label_mapping = {
    "human_submissions_romantic_relationships": "Human-Human\nSubmissions Romantic",
    "human_submissions_non_romantic_relationships": "Human-Human\nSubmissions Non-\nRomantic",
    "human_ai_chatlogs_USER":    "Human-AI Chatlogs\nUser",
    "human_ai_chatlogs_Chatbot": "Human-AI Chatlogs\nChatbot",
    "human_submissions":         "Human-Human Submissions",
    "human_AI_submissions":      "Human-AI \nSubmissions"
}
# pastel-but-vibrant palette for density lines
pastel_colors = [
    '#FF6B6B',  # pastel red
    '#A09DDC',  # pastel lavender
    '#88B04B',  # pastel green
    '#FFE066',  # pastel yellow
    '#5AD3D1',  # pastel turquoise
    '#FF9AA2',  # pastel pink
    '#FFB347',  # pastel orange
    '#C17BA7',  # pastel magenta
    '#6FA8DC',  # pastel blue
    '#B4D2BA',  # pastel mint
]

def parse_args():
    p = argparse.ArgumentParser(description="Plot score density curves for CSV files.")
    p.add_argument("--input_dir",       required=True,
                   help="Directory containing CSV files.")
    p.add_argument("--output_file",     required=True,
                   help="Path (including filename) for the saved PDF/PNG.")
    p.add_argument("--exclude_subreddits", default="",
                   help="Comma‑separated subreddits to drop before grouping.")
    p.add_argument("--pattern",         default="*.csv",
                   help="Glob pattern to match CSV files.")
    return p.parse_args()

def main():
    args = parse_args()
    exclude = {s.strip() for s in args.exclude_subreddits.split(',') if s.strip()}

    # font settings
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'legend.title_fontsize': 16
    })

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching {args.pattern} in {args.input_dir}")

    # collect score series for each bar
    distributions = {}
    for path in paths:
        df = pd.read_csv(path, index_col=False)
        basename = os.path.splitext(os.path.basename(path))[0]

        # optional subreddit filtering
        if exclude and 'subreddit' in df.columns:
            df = df[~df['subreddit'].isin(exclude)]

        if basename == "human_submissions":
            for group_name, subs in [
                ("romantic_relationships", romantic_relationships),
                ("non_romantic_relationships", non_romantic_relationships)
            ]:
                grp = df[df['subreddit'].isin(subs)]
                distributions[f"{basename}_{group_name}"] = grp['score'].dropna()

        elif basename == "human_ai_chatlogs":
            if 'party' not in df.columns:
                raise KeyError("Expected a 'party' column in human_ai_chatlogs_emo.csv")
            for party in ["USER", "Chatbot"]:
                grp = df[df['party'] == party]
                distributions[f"{basename}_{party}"] = grp['score'].dropna()

        else:
            distributions[basename] = df['score'].dropna()

    # plot densities
    fig, ax = plt.subplots(figsize=(12, 7))
    for idx, (label, series) in enumerate(distributions.items()):
        color = pastel_colors[idx % len(pastel_colors)]
        series.plot(
            kind='kde',
            ax=ax,
            label=label_mapping.get(label, label),
            color=color,
            linewidth=2
        )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("")
    ax.legend(title="Group", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    # save PDF and PNG
    plt.savefig(args.output_file)
    base, _ = os.path.splitext(args.output_file)
    plt.savefig(f"{base}.png")
    plt.close()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plot_sfw_nsfw.py

Reads all CSV files in a folder, each with columns:
  - label:  "NSFW" or "SFW"
  - score:  a numeric score (e.g., probability)

Splits into bars exactly as before:
  - human_submissions_emo → two bars (romantic vs non‑romantic subreddits)
  - human_ai_chatlogs_emo → two bars (USER vs Chatbot)
  - all others → one bar per file

For each bar, sums `score` by `label` and plots a stacked bar of NSFW vs. SFW proportions.
Exports both PDF and PNG.
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# relationship groups (unchanged)
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

# two labels and their colors
label_values = ["NSFW", "SFW"]
color_order = ['#FF6B6B', '#51CF66']  # NSFW=red, SFW=green

# x‑tick label mapping (same keys as before)
label_mapping = {
    "human_submissions_romantic_relationships": "Human-Human\nSubmissions Romantic",
    "human_submissions_non_romantic_relationships": "Human-Human\nSubmissions Non-\nRomantic",
    "human_ai_chatlogs_USER":    "Human-AI Chatlogs\nUser",
    "human_ai_chatlogs_Chatbot": "Human-AI Chatlogs\nChatbot",
    "human_submissions":         "Human-Human Submissions",
    "human_AI_submissions":      "Human-AI \nSubmissions"
}

def parse_args():
    p = argparse.ArgumentParser(description="Plot NSFW vs SFW proportions per file/group.")
    p.add_argument("--input_dir",       required=True,
                   help="Folder containing your CSV files.")
    p.add_argument("--output_file",     required=True,
                   help="Output filename (will save .pdf and .png).")
    p.add_argument("--exclude_subreddits", default="",
                   help="Comma‑separated subreddits to drop before grouping.")
    p.add_argument("--pattern",         default="*.csv",
                   help="Glob pattern for CSVs (default '*.csv').")
    return p.parse_args()

def main():
    args = parse_args()
    exclude = {s.strip() for s in args.exclude_subreddits.split(',') if s.strip()}

    # bump up font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
        'legend.title_fontsize': 18
    })

    paths = glob.glob(os.path.join(args.input_dir, args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files in {args.input_dir} matching {args.pattern}")

    proportions = {}

    for path in sorted(paths):
        df = pd.read_csv(path, index_col=False)

        # ensure we have the two columns
        if 'label' not in df.columns or 'score' not in df.columns:
            raise KeyError(f"{os.path.basename(path)} must have columns 'label' and 'score'")

        basename = os.path.splitext(os.path.basename(path))[0]

        # optional subreddit filter
        if exclude and 'subreddit' in df.columns:
            df = df[~df['subreddit'].isin(exclude)]

        # SPECIAL: human_submissions_emo → romantic vs non‑romantic
        if basename == "human_submissions":
            for group_name, subs in [
                ("romantic_relationships", romantic_relationships),
                ("non_romantic_relationships", non_romantic_relationships)
            ]:
                grp = df[df['subreddit'].isin(subs)]
                sums = grp.groupby('label')['score'].sum().reindex(label_values, fill_value=0)
                props = sums / sums.sum() if sums.sum() > 0 else sums
                proportions[f"{basename}_{group_name}"] = props

        # SPECIAL: human_ai_chatlogs_emo → USER vs Chatbot
        elif basename == "human_ai_chatlogs":
            if 'party' not in df.columns:
                raise KeyError("Expected a 'party' column in human_ai_chatlogs_emo.csv")
            for party in ["USER", "Chatbot"]:
                grp = df[df['party'] == party]
                sums = grp.groupby('label')['score'].sum().reindex(label_values, fill_value=0)
                props = sums / sums.sum() if sums.sum() > 0 else sums
                proportions[f"{basename}_{party}"] = props

        # DEFAULT: one bar per file
        else:
            sums = df.groupby('label')['score'].sum().reindex(label_values, fill_value=0)
            props = sums / sums.sum() if sums.sum() > 0 else sums
            proportions[basename] = props

    # build DataFrame and plot
    prop_df = pd.DataFrame(proportions).T

    fig, ax = plt.subplots(figsize=(12, 7))
    prop_df.plot(
        kind='bar',
        stacked=True,
        width=0.8,
        color=color_order,
        ax=ax
    )

    # simplify x‑tick labels
    new_labels = [label_mapping.get(lbl, lbl) for lbl in prop_df.index]
    ax.set_xticklabels(new_labels, rotation=45, ha='right')

    ax.set_title("")       # no title
    ax.set_xlabel("")      # no x label
    ax.set_ylabel("Proportion")  # y label

    # legend to the right
    ax.legend(title='Label', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    # save both formats
    plt.savefig(args.output_file)
    base, _ = os.path.splitext(args.output_file)
    plt.savefig(f"{base}.png")
    plt.close()

if __name__ == "__main__":
    main()

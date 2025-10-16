#!/usr/bin/env python3
"""
plot_emotions.py

A script to read all CSV files in a specified folder, compute the emotion distributions,
and produce a stacked bar chart where each bar corresponds to one file.

Special cases:
  - human_submissions_emo.csv → two bars for romantic vs non-romantic relationships
  - human_ai_chatlogs_emo.csv → two bars for USER vs Chatbot turns
  - all other CSVs → one bar per file
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

# emotion color mapping (ordered by emotion_cols index), more vibrant pastels:
color_order = [
    '#FF6B6B',  # anger       – warm pastel red
    '#D08510',  # disgust     – rich amber
    '#A162F7',  # fear        – vivid pastel purple
    '#B0B0B0',  # pessimism   – medium gray
    '#4C6EF5',  # sadness     – bright pastel blue
    '#4DD0E1',  # anticipation– clear pastel cyan
    '#FFE066',  # surprise    – sunny pastel yellow
    '#FFA94D',  # optimism    – warm pastel orange
    '#51CF66',  # joy         – lively pastel green
    '#FF65A3',  # love        – bold pastel pink
    '#B2F2BB'   # trust       – fresh pastel mint
]


# x‑tick label mapping
label_mapping = {
    "human_submissions_emo_romantic_relationships": "Human-Human\nSubmissions \nRomantic",
    "human_submissions_emo_non_romantic_relationships": "Human-Human\nSubmissions \nNon-Romantic",
    "human_ai_chatlogs_emo_USER":    "Human-AI Chatlogs\nUser",
    "human_ai_chatlogs_emo_Chatbot": "Human-AI Chatlogs\nChatbot",
    "human_submissions_emo":         "Human-Human Submissions",
    "human_AI_submissions_emo":      "Human-AI \nSubmissions"
}

def parse_args():
    p = argparse.ArgumentParser(description="Plot stacked emotion bars for CSV files.")
    p.add_argument("--input_dir",       required=True, help="Directory containing CSV files.")
    p.add_argument("--output_file",     required=True, help="Path for the saved PDF figure.")
    p.add_argument("--exclude_subreddits", default="",
                   help="Comma‑separated subreddits to drop before plotting.")
    p.add_argument("--pattern",         default="*.csv",
                   help="Glob pattern to match CSV files (default '*.csv').")
    return p.parse_args()

def main():
    args = parse_args()
    emotion_cols = [
        'anger','disgust','fear','pessimism','sadness',
        'anticipation','surprise','optimism','joy','love','trust'
    ]
    exclude = {s.strip() for s in args.exclude_subreddits.split(',') if s.strip()}

    # enlarge fonts globally
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,
        'legend.title_fontsize': 16
    })

    paths = glob.glob(os.path.join(args.input_dir, args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {args.pattern} in {args.input_dir}")

    proportions = {}

    for path in sorted(paths):
        df = pd.read_csv(path, index_col=False)
        basename = os.path.splitext(os.path.basename(path))[0]

        # optional subreddit filtering
        if exclude and 'subreddit' in df.columns:
            df = df[~df['subreddit'].isin(exclude)]

        if basename == "human_submissions_emo":
            # two bars: romantic vs non-romantic
            for group_name, subs in [
                ("romantic_relationships", romantic_relationships),
                ("non_romantic_relationships", non_romantic_relationships)
            ]:
                grp = df[df['subreddit'].isin(subs)]
                sums = grp[emotion_cols].sum()
                props = sums / sums.sum() if sums.sum() > 0 else sums
                proportions[f"{basename}_{group_name}"] = props

        elif basename == "human_ai_chatlogs_emo":
            # two bars: USER vs Chatbot (column is lowercase 'party')
            if 'party' not in df.columns:
                raise KeyError("Expected a 'party' column in human_ai_chatlogs_emo.csv")
            for party in ["USER", "Chatbot"]:
                grp = df[df['party'] == party]
                sums = grp[emotion_cols].sum()
                props = sums / sums.sum() if sums.sum() > 0 else sums
                proportions[f"{basename}_{party}"] = props

        else:
            # default: one bar per file
            sums = df[emotion_cols].sum()
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

    # remove plot title and x-axis label
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("Proportion of Emotions")

    # place legend outside to the right
    ax.legend(title='Emotion', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(args.output_file)
    plt.close()

if __name__ == "__main__":
    main()

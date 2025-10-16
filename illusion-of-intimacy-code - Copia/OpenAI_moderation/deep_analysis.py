#!/usr/bin/env python3
"""
first_spike_by_category_summary.py

For each category in ["self-harm", "sexual", "violence", "harassment"]:
 1. Find the first turn where that category’s score exceeds 0.5.
 2. Count how many dialogues spike at each turn 1–6.
 3. For turns 1–4, break down those first spikes by USER vs Chatbot.
Writes summary to first_spike_by_category_summary.txt.
"""

import pandas as pd

# 1. Configuration
csv_path   = "/home/mhchu/AI-Companion/human-ai/OpenAI_moderation/results/human_ai_chatlogs.csv"
output_txt = "first_spike_by_category_summary.txt"
threshold  = 0.5
categories = ["self-harm", "sexual", "violence", "harassment"]

# 2. Load data
df = pd.read_csv(csv_path)

# 3. Compute and write summary
with open(output_txt, "w") as f:
    for cat in categories:
        f.write(f"=== Category: {cat} (threshold > {threshold}) ===\n")
        spikes = []
        # collect first-spike info per dialogue
        for dlg_id, grp in df.groupby("id"):
            spike_rows = grp[grp[cat] > threshold]
            if spike_rows.empty:
                continue
            first_turn = spike_rows["turn"].min()
            party = spike_rows.loc[spike_rows["turn"] == first_turn, "party"].iloc[0]
            spikes.append({"turn": first_turn, "party": party})

        if not spikes:
            f.write("No dialogues with a first spike above threshold.\n\n")
            continue

        spike_df = pd.DataFrame(spikes)

        # 3a. Counts of first spikes by turn (only turns 1–6)
        counts_by_turn = (
            spike_df[spike_df["turn"].between(1, 6)]
            .turn
            .value_counts()
            .reindex(range(1, 7), fill_value=0)
            .sort_index()
        )
        f.write("Counts of first spikes by turn (1–6):\n")
        f.write(counts_by_turn.to_string() + "\n")

        # 3b. Counts of first spikes by party for turns 1–4
        party_counts = (
            spike_df[spike_df["turn"].between(1, 4)]
            .groupby(["turn", "party"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=range(1, 5), fill_value=0)
        )
        # ensure both USER and Chatbot columns exist
        for role in ["USER", "Chatbot"]:
            if role not in party_counts.columns:
                party_counts[role] = 0
        party_counts = party_counts[["USER", "Chatbot"]]

        f.write("Counts of first spikes by party (turns 1–4):\n")
        f.write(party_counts.to_string() + "\n\n")

print(f"✅ Summary written to {output_txt}")

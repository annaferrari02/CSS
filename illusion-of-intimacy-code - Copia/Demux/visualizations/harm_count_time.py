import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

# 0. Increase fonts
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'legend.title_fontsize': 22
})

# 1. Load data (with party)
df_scores = pd.read_csv(
    "/home/mhchu/AI-Companion/human-ai/data/data/combined_data.csv",
    dtype={"id": str}
)
df_meta = pd.read_csv(
    "/home/mhchu/AI-Companion/human-ai/data/data/human_AI_submissions.csv",
    usecols=["id", "timestamp"],
    dtype={"id": str},
    low_memory=False
)
df_meta["timestamp"] = pd.to_datetime(
    df_meta["timestamp"].astype(float), unit="s", errors="coerce"
)
df_meta = df_meta.dropna(subset=["timestamp"])
df = pd.merge(df_scores, df_meta, on="id", how="inner").set_index("timestamp")

# 2. Flag harm spikes (>0.5), excluding 'hate'
harm_cols = ["harassment", "sexual", "self-harm", "violence"]
spikes = df[harm_cols].gt(0.5).astype(int)
spikes["party"] = df["party"]

# 3. Calculate dialogue counts per month by party
dialogues_per_month = df.groupby([pd.Grouper(freq="M"), "party"]).size().unstack("party").fillna(0)

# 4. Monthly harm counts by category & party
monthly_harm_counts = (
    spikes
    .groupby([pd.Grouper(freq="M"), "party"])[harm_cols]
    .sum()
    .unstack("party")
    .fillna(0)
)

# 5. Normalize: Divide harm counts by number of dialogues to get rate per dialogue
# Handle any division by zero by replacing with 0
normalized_counts = pd.DataFrame()
for cat in harm_cols:
    for party in ["USER", "Chatbot"]:
        if party in monthly_harm_counts[cat].columns and party in dialogues_per_month.columns:
            if (cat, party) in monthly_harm_counts.columns:
                normalized_counts[(cat, party)] = monthly_harm_counts[(cat, party)].div(
                    dialogues_per_month[party], axis=0
                ).fillna(0)

# 6. Colors (without hate)
harm_colors = {
    "harassment": "#FF6B6B",
    "sexual":      "#D08510",
    "self-harm":   "#B0B0B0",
    "violence":    "#4C6EF5"
}

# 7. Plot
fig, ax = plt.subplots(figsize=(10, 5.5))
for cat in harm_cols:
    if (cat, "USER") in normalized_counts.columns:
        ax.plot(
            normalized_counts.index,
            normalized_counts[(cat, "USER")],
            color=harm_colors[cat],
            linewidth=3,
            linestyle='-'
        )
    if (cat, "Chatbot") in normalized_counts.columns:
        ax.plot(
            normalized_counts.index,
            normalized_counts[(cat, "Chatbot")],
            color=harm_colors[cat],
            linewidth=3,
            linestyle='--'
        )

# 8. Labels
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("Harm Spikes per Dialogue", labelpad=10)

# 9. Build handles for legends (without hate)
color_handles = [Line2D([0], [0], color=harm_colors[c], lw=4) for c in harm_cols]
style_handles = [
    Line2D([0], [0], color="black", lw=4, ls='-'),
    Line2D([0], [0], color="black", lw=4, ls='--')
]

# 10. Place legends with border - ADJUSTED POSITIONS
fig.legend(
    handles=color_handles,
    labels=harm_cols,
    title="Harm Category",
    loc="upper left",
    bbox_to_anchor=(0.75, 0.90),
    frameon=True,
    edgecolor='black'
)
fig.legend(
    handles=style_handles,
    labels=["User", "Chatbot"],
    title="Party",
    loc="upper left",
    bbox_to_anchor=(0.75, 0.42),
    frameon=True,
    edgecolor='black'
)

# 11. Adjust layout so nothing is clipped
fig.subplots_adjust(right=0.70, top=0.90, bottom=0.10)

# 12. Format x-axis ticks every 3 months as MM/YYYY
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_ha('right')

# 13. Grid
ax.grid(True)

# 14. Save
out_dir = "/home/mhchu/AI-Companion/human-ai/Demux/emo_time"
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "harm_spikes_normalized_by_dialogue_mmYYYY.png"),
            dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, "harm_spikes_normalized_by_dialogue_mmYYYY.pdf"),
            bbox_inches='tight')
plt.close(fig)
import pandas as pd
import matplotlib.pyplot as plt
import os

# 0. Enlarge fonts globally
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16
})

# 1. Load all scores (cast id → str)
df_em = pd.read_csv(
    "/home/mhchu/AI-Companion/human-ai/data/data/combined_data.csv",
    dtype={"id": str}
)

# 2. Load metadata (timestamps + party)
df_meta = pd.read_csv(
    "/home/mhchu/AI-Companion/human-ai/data/data/human_AI_submissions.csv",
    usecols=["id", "timestamp"],
    dtype={"id": str},
    low_memory=False
)
df_meta["timestamp"] = pd.to_datetime(
    df_meta["timestamp"].astype(float),
    unit="s",
    errors="coerce"
)
df_meta = df_meta.dropna(subset=["timestamp"])

# 3. Merge and set timestamp index
df = pd.merge(df_em, df_meta, on="id", how="inner")
df = df.set_index("timestamp")

# 4. Define your groups of columns
groups = {
    "Emotions": [
        'anger', 'disgust', 'fear', 'sadness',
        'surprise', 'joy', 'love', 'optimism'
    ],
    "Moral Foundations": [
        "Purity", "Thin Morality", "Authority",
        "Equality", "Loyalty", "Care", "Proportionality"
    ],
    "Harm": [
        "harassment", "sexual", "hate", "self-harm", "violence"
    ]
}

# 5. Updated vibrant palette for Emotions
emotion_colors = {
    'anger':    '#E63946',
    'disgust':  '#8AC926',
    'fear':     '#845EC2',
    'sadness':  '#264653',
    'surprise': '#FFB30F',
    'joy':      '#FFD166',
    'love':     '#FF5D9E',
    'optimism': '#06D6A0'
}

# 6. Default palette for other groups
color_order = [
    '#FF6B6B', '#D08510', '#A162F7', '#B0B0B0', '#4C6EF5',
    '#4DD0E1', '#FFE066', '#FFA94D', '#51CF66', '#FF65A3', '#B2F2BB'
]

# 7. Output directory
out_dir = "/home/mhchu/AI-Companion/human-ai/Demux/emo_time"
os.makedirs(out_dir, exist_ok=True)

# 8. Loop over each group and each party
for name, cols in groups.items():
    safe = name.lower().replace(" ", "_")
    # choose colors
    if name == "Emotions":
        colors = [emotion_colors[c] for c in cols]
    else:
        colors = color_order[:len(cols)]

    for party in ["USER", "Chatbot"]:
        # a) subset by party
        df_p = df[df["party"] == party]
        # b) compute dominant category per row
        dom = df_p[cols].idxmax(axis=1)
        # c) count per calendar‐month and reindex
        counts = (
            dom
            .groupby(pd.Grouper(freq="ME"))
            .value_counts()
            .unstack(fill_value=0)
            .reindex(columns=cols, fill_value=0)
        )
        # d) convert to proportions
        prop = counts.div(counts.sum(axis=1), axis=0)

        # e) plot
        fig, ax = plt.subplots(figsize=(10, 5))
        prop.plot(
            kind="bar",
            stacked=True,
            color=colors,
            width=0.8,
            ax=ax
        )

        # f) format x‐ticks as MM/YY
        months = prop.index.to_period("M").to_timestamp()
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(
            [dt.strftime("%m/%y") for dt in months],
            rotation=45,
            ha="right"
        )

        # g) labels & legend
        ax.set_xlabel("")
        ax.set_ylabel(f"{name} ({party})")
        ax.legend(title=name, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        # h) save outputs
        out_png = os.path.join(out_dir, f"{safe}_{party.lower()}.png")
        out_pdf = os.path.join(out_dir, f"{safe}_{party.lower()}.pdf")
        fig.savefig(out_png, dpi=300)
        fig.savefig(out_pdf)
        plt.close(fig)

#!/usr/bin/env python3
"""
produce_radar.py

Usage:
    python produce_radar.py human_ai_chatlogs_emo.csv /path/to/output_dir
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot dominant-emotion radar for human–AI chatlogs"
    )
    p.add_argument(
        'input_csv',
        help="Path to human_ai_chatlogs_emo.csv"
    )
    p.add_argument(
        'output_path',
        help="Directory where the radar plots (PDF & PNG) will be saved"
    )
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # 1. Load data
    df = pd.read_csv(args.input_csv)

    # 2. Define emotion columns (removed 'pessimism') and parties
    emotion_cols = [
        'anger', 'disgust', 'fear', 'sadness',
        'surprise', 'optimism',
        'joy', 'love'
    ]
    parties = ['USER', 'Chatbot']

    # 3. Determine dominant emotion per row
    df['dominant'] = df[emotion_cols].idxmax(axis=1)

    # 4. Compute frequency‐based proportions for each party
    proportions = {}
    for party in parties:
        sub = df[df['party'] == party]
        counts = sub['dominant'].value_counts().reindex(emotion_cols, fill_value=0)
        proportions[party] = counts / counts.sum()

    # 5. Radar‐plot setup
    labels = emotion_cols
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    # 6. Increase global font sizes - UPDATED with bigger font sizes
    plt.rcParams.update({
        'font.size': 24,              # Increased from 16
        'xtick.labelsize': 22,        # Increased from 16
        'legend.fontsize': 20,        # Increased from 14
        'axes.titlesize': 26,         # Added title size
        'figure.titlesize': 28        # Added figure title size
    })

    # 7. Create figure
    fig, ax = plt.subplots(figsize=(9, 6), subplot_kw=dict(polar=True))  # Increased figure size

    # 8. More vibrant pastel colors
    colors = {
        'USER':    {'line': '#FF6B6B', 'fill': '#FF6B6B'},  # vibrant pastel red
        'Chatbot': {'line': '#4DD0E1', 'fill': '#4DD0E1'}   # vibrant pastel cyan
    }

    # 9. Plot each party
    for party in parties:
        vals = proportions[party].tolist()
        vals += vals[:1]
        ax.plot(
            angles, vals,
            color=colors[party]['line'],
            linewidth=5,              # Increased from 3 for much better visibility
            label=party
        )
        ax.fill(
            angles, vals,
            color=colors[party]['fill'],
            alpha=0
        )

    # 10. Ticks & label padding
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', pad=35)   # Significantly increased padding to move labels further from circle
    ax.set_yticklabels([])             # hide radial labels

    # 11. Legend to the right with increased spacing
    ax.legend(loc='center left', bbox_to_anchor=(1.35, 0.5), fontsize=20)  # Explicit fontsize and adjusted position

    plt.tight_layout(pad=2.0)  # Increased padding for better spacing

    # 12. Save to PDF & PNG with higher DPI for sharper text
    base = os.path.splitext(os.path.basename(args.input_csv))[0]
    pdf_out = os.path.join(args.output_path, f"{base}_radar.pdf")
    png_out = os.path.join(args.output_path, f"{base}_radar.png")
    plt.savefig(pdf_out, dpi=300)
    plt.savefig(png_out, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
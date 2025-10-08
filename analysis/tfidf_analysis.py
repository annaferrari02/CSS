import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool, cpu_count

# Import from shared preprocessing module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from text_cleaner import clean_text


def run_tfidf_analysis(data_path, output_dir=None, use_multiprocessing=False):
    """
    Run TF-IDF analysis comparing USER and Chatbot vocabulary
    
    Args:
        data_path: Path to CSV with 'party' and 'text' columns
        output_dir: Directory to save results (default: current directory)
        use_multiprocessing: Whether to use parallel processing for text cleaning
    """
    # 🔹 imposta la directory di output
    if output_dir is None:
        output_dir = Path("output")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_dir.resolve()}")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, usecols=["party", "text"])
    texts = df["text"].astype(str).tolist()
    
    # Clean text
    print("Cleaning text...")
    if use_multiprocessing:
        with Pool(cpu_count()) as pool:
            df["clean_text"] = pool.map(clean_text, texts)
    else:
        df["clean_text"] = df["text"].astype(str).apply(clean_text)
    
    # Split USER vs Chatbot
    usr = df.loc[df.party == "USER", "clean_text"].tolist()
    bot = df.loc[df.party == "Chatbot", "clean_text"].tolist()
    
    print(f"USER messages: {len(usr)}")
    print(f"Chatbot messages: {len(bot)}")
    
    # TF-IDF
    print("Computing TF-IDF...")
    vec_u = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
    vec_b = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
    X_u = vec_u.fit_transform(usr).toarray()
    X_b = vec_b.fit_transform(bot).toarray()
    
    mu_u = X_u.mean(axis=0)
    mu_b = X_b.mean(axis=0)
    terms_u = vec_u.get_feature_names_out()
    terms_b = vec_b.get_feature_names_out()
    common = set(terms_u).intersection(terms_b)
    
    print(f"Common terms: {len(common)}")
    
    diffs = [
        (t, mu_b[np.where(terms_b == t)[0][0]] - mu_u[np.where(terms_u == t)[0][0]])
        for t in common
    ]
    
    # Top 20 each
    top_user = sorted(diffs, key=lambda x: x[1])[:20]  # Most negative diffs
    top_bot = sorted(diffs, key=lambda x: x[1], reverse=True)[:20]  # Most positive diffs
    
    # Create visualization
    print("Creating visualization...")
    n = 20
    spacing = 6
    
    y_bot = np.arange(n, 2*n)[::-1] * spacing
    y_user = np.arange(n)[::-1] * spacing
    
    chatbot_labels = [t for t, _ in top_bot]
    chatbot_vals = [v for _, v in top_bot]
    user_labels = [t for t, _ in top_user[::-1]]
    user_vals = [v for _, v in top_user[::-1]]
    
    all_labels = chatbot_labels + user_labels
    all_vals = chatbot_vals + user_vals
    
    colors = {
        'USER': {'line': '#FF6B6B', 'fill': '#FF6B6B'},
        'Chatbot': {'line': '#4DD0E1', 'fill': '#4DD0E1'}
    }
    
    fig, ax = plt.subplots(figsize=(6, 8))
    
    ax.barh(y_bot, all_vals[:n],
            height=3,
            color=colors['Chatbot']['fill'],
            edgecolor=colors['Chatbot']['line'],
            label="Chatbot")
    
    ax.barh(y_user, all_vals[n:],
            height=3,
            color=colors['USER']['fill'],
            edgecolor=colors['USER']['line'],
            label="USER")
    
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_yticks(np.concatenate([y_bot, y_user]))
    ax.set_yticklabels(all_labels, fontsize=13)
    ax.tick_params(axis="y", pad=15)
    ax.set_xlabel("Delta TF-IDF (Chatbot - User)", fontsize=15)
    
    leg = ax.legend(loc="lower right", fontsize=13, frameon=True)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)
    
    plt.tight_layout()
    
    # 🔹 salva nella cartella "output"
    for ext in ("png", "pdf"):
        output_file = output_dir / f"tfidf_user_vs_bot.{ext}"
        fig.savefig(output_file, dpi=300)
        print(f"  ✅ Saved: {output_file}")
    
    plt.close()
    
    return diffs, top_user, top_bot


if __name__ == "__main__":
    DATA_PATH = Path("data") / "human_ai_chatlogs.csv"
    
    # Fallback paths
    if not DATA_PATH.exists():
        possible_paths = [
            Path("C:/Users/anna2/OneDrive/Desktop/CSS/data/human_ai_chatlogs.csv"),
            Path("/home/mhchu/AI-Companion/human-ai/data/data/human_ai_chatlogs.csv")
        ]
        for path in possible_paths:
            if path.exists():
                DATA_PATH = path
                print(f"Using legacy path: {DATA_PATH}")
                break
    
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found at {DATA_PATH}")
        sys.exit(1)
    
    # 🔹 run analysis with output in "output/"
    diffs, top_user, top_bot = run_tfidf_analysis(DATA_PATH, output_dir="output", use_multiprocessing=False)


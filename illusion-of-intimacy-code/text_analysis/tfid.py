#!/usr/bin/env python3
import os
import re
import pandas as pd
import nltk
from multiprocessing import Pool, cpu_count
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt

# 1) Data path
DATA_PATH = "/home/mhchu/AI-Companion/human-ai/data/data/human_ai_chatlogs.csv"

# 2) NLTK setup (only need to run once)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
extra = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","any","both","each","few","more",
    "most","other","some","such","no","nor","not","only","own","same","so",
    "than","too","very","s","t","can","will","just","don","should","now",
    "dont","ill","im"
}
stopwords_list = stop_words.union(extra)
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    t = re.sub(r'http\S+|www\S+|t\.co\S+', '', text)
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip().lower()
    toks = word_tokenize(t)
    return " ".join(
        lemmatizer.lemmatize(tok)
        for tok in toks
        if tok and tok not in stopwords_list
    )

def main():
    # Load & clean in parallel
    df = pd.read_csv(DATA_PATH, usecols=["party","text"])
    texts = df["text"].astype(str).tolist()
    with Pool(cpu_count()) as pool:
        df["clean_text"] = pool.map(clean_text, texts)

    # Split USER vs Chatbot
    usr = df.loc[df.party=="USER", "clean_text"].tolist()
    bot = df.loc[df.party=="Chatbot", "clean_text"].tolist()

    # TF-IDF
    vec_u = TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words="english")
    vec_b = TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words="english")
    X_u = vec_u.fit_transform(usr).toarray()
    X_b = vec_b.fit_transform(bot).toarray()

    mu_u = X_u.mean(axis=0)
    mu_b = X_b.mean(axis=0)
    terms_u = vec_u.get_feature_names_out()
    terms_b = vec_b.get_feature_names_out()
    common  = set(terms_u).intersection(terms_b)

    diffs = [
        (t,
         mu_b[np.where(terms_b == t)[0][0]]
         - mu_u[np.where(terms_u == t)[0][0]])
        for t in common
    ]

    # Top 20 each
    top_user = sorted(diffs, key=lambda x: x[1])[:20]              # Most negative diffs
    top_bot  = sorted(diffs, key=lambda x: x[1], reverse=True)[:20]  # Most positive diffs

    # Positions with increased vertical spacing
    n = 20
    spacing = 6

    # Reverse both halves:
    # Chatbot on top half (longest→shortest)
    y_bot  = np.arange(n, 2*n)[::-1] * spacing
    # USER on bottom half (shortest→longest)
    y_user = np.arange(n)[::-1] * spacing

    # Labels & values
    chatbot_labels = [t for t,_ in top_bot]
    chatbot_vals   = [v for _,v in top_bot]
    user_labels    = [t for t,_ in top_user[::-1]]
    user_vals      = [v for _,v in top_user[::-1]]

    all_labels = chatbot_labels + user_labels
    all_vals   = chatbot_vals   + user_vals

    # Colors
    colors = {
        'USER':    {'line': '#FF6B6B', 'fill': '#FF6B6B'},
        'Chatbot': {'line': '#4DD0E1', 'fill': '#4DD0E1'}
    }

    # Plot
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

    ax.set_xlabel("Δ TF-IDF (Chatbot − User)", fontsize=15)

    # Framed legend
    leg = ax.legend(loc="lower right", fontsize=13, frameon=True)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"tfidf_user_vs_bot.{ext}", dpi=300)
    print("✅ Saved tfidf_user_vs_bot.{png,pdf}")

if __name__ == "__main__":
    main()

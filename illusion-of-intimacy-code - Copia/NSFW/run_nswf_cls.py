#!/usr/bin/env python
"""
Run the michellejieli/NSFW_text_classifier on a CSV of texts
with custom cleaning in parallel (32 processes), on a single GPU (cuda:4),
batch size = 64, and output label + score to a CSV in a given folder,
preserving the input filename.
"""

import os
import re
import argparse
import pandas as pd
import torch
import nltk
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
from tqdm import tqdm

# ── 1) AUTHENTICATE ─────────────────────────────────────────────────────────
HF_API_KEY = ""  # ← insert your HF API key here
login(token=HF_API_KEY)

# ── 2) SET DEVICE ──────────────────────────────────────────────────────────
DEVICE_ID = 4
device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
print(f"Using device = {device}")

# ── 3) NLTK SETUP ───────────────────────────────────────────────────────────
nltk.download("punkt",    quiet=True)
nltk.download("stopwords",quiet=True)
nltk.download("wordnet",   quiet=True)
lemmatizer     = WordNetLemmatizer()
stop_words     = set(stopwords.words("english"))
stopwords_list = [
    "i","me","my","myself","we","our","ours","ourselves",
    "you","your","yours","yourself","yourselves","he","him",
    "his","himself","she","her","hers","herself","it","its",
    "itself","they","them","their","theirs","themselves","what",
    "which","who","whom","this","that","these","those","am",
    "is","are","was","were","be","been","being","have","has",
    "had","having","do","does","did","doing","a","an","the",
    "and","but","if","or","because","as","until","while","of",
    "at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to",
    "from","up","down","in","out","on","off","over","under",
    "again","further","then","once","here","there","when",
    "where","why","how","all","any","both","each","few","more",
    "most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","s","t","can","will",
    "just","don","should","now"
]

def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML, remove punctuation, lemmatize & drop stopwords."""
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+|www\S+|t\.co\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\/?[ru]\/\w+", "", text)
    text = re.sub(r"&gt;|&lt;|&amp;", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    tokens = word_tokenize(text)
    return " ".join(
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and tok not in stopwords_list
    )

def main():
    parser = argparse.ArgumentParser(
        description="Classify CSV texts with NSFW_text_classifier on GPU 4."
    )
    parser.add_argument(
        "input_csv",
        help="Path to input CSV (must have a 'text' column)"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write the output CSV into (preserves filename)"
    )
    parser.add_argument(
        "--model",
        default="michellejieli/NSFW_text_classifier",
        help="HF model repo ID"
    )
    args = parser.parse_args()

    # ── 4) LOAD CSV & FILTER ──────────────────────────────────────────────────
    df = pd.read_csv(args.input_csv)
    df = df[df["text"].notna() & df["text"].str.strip().ne("")].reset_index(drop=True)

    # ── 5) CLEAN TEXT IN PARALLEL ─────────────────────────────────────────────
    texts = df["text"].tolist()
    with Pool(processes=32) as pool:
        cleaned = list(tqdm(pool.imap(clean_text, texts),
                            total=len(texts),
                            desc="Cleaning texts"))
    df["cleaned_text"] = cleaned

    # ── 6) LOAD TOKENIZER & MODEL ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    # ── 7) BUILD PIPELINE ────────────────────────────────────────────────────
    classifier = pipeline(
        task="sentiment-analysis",      # for NSFW_text_classifier
        model=model,
        tokenizer=tokenizer,
        device=DEVICE_ID
    )

    # ── 8) INFERENCE (batch_size=64) ──────────────────────────────────────────
    all_results = []
    batch_size  = 64
    for i in tqdm(range(0, len(cleaned), batch_size), desc="Classifying"):
        batch = cleaned[i : i + batch_size]
        all_results.extend(classifier(batch, truncation=True, batch_size=batch_size))

    df["label"] = [r["label"] for r in all_results]
    df["score"] = [r["score"] for r in all_results]

    # ── 9) WRITE OUTPUT ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    basename    = os.path.basename(args.input_csv)
    output_path = os.path.join(args.output_dir, basename)
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
# 0) FORCE OpenBLAS/Numexpr → use 1 thread only
os.environ["OPENBLAS_NUM_THREADS"  ] = "1"
os.environ["MKL_NUM_THREADS"       ] = "1"
os.environ["NUMEXPR_NUM_THREADS"   ] = "1"

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap
import matplotlib.pyplot as plt
from pathlib import Path

# 1) Data
# Ottieni la directory corrente dello script e risali di un livello
base_dir = Path(__file__).resolve().parent.parent  # rimuove "analysis"

# Costruisci il percorso corretto
DATA_PATH = base_dir / "data" / "human_ai_chatlogs_ilmr.csv" #MODIFICARE 

# 🔹 Carica il dataset
df = pd.read_csv(DATA_PATH, usecols=["party", "text"])
texts = df["text"].astype(str).tolist()
parties = df["party"].tolist()

# 2) Load E5 embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer("intfloat/e5-base", device=device)

# 3) Embed all turns
embs = model.encode(
    texts,
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True
)

# 4) Centroid similarity
mask_u = np.array(parties)=="USER"
mask_b = np.array(parties)=="Chatbot"
cent_u = embs[mask_u].mean(axis=0, keepdims=True)
cent_b = embs[mask_b].mean(axis=0, keepdims=True)
cent_sim = cosine_similarity(cent_u, cent_b)[0,0]
print(f"Overall USER⇆Chatbot centroid cosine: {cent_sim:.4f}")

# 5) UMAP projection
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)
proj = reducer.fit_transform(embs)

# 6) Plot
plt.figure(figsize=(8,6))
colors = {"USER":"#FF5555", "Chatbot":"#5555FF"}
for label in ["USER","Chatbot"]:
    idx = np.where(np.array(parties)==label)
    plt.scatter(
        proj[idx,0], proj[idx,1],
        s=5, alpha=0.6,
        c=colors[label],
        label=label
    )

plt.xticks([],[])
plt.yticks([],[])
plt.xlabel("UMAP 1", fontsize=14)
plt.ylabel("UMAP 2", fontsize=14)
plt.legend(loc="lower right", fontsize=12, frameon=False)
plt.tight_layout()

# 7) Save
# crea la cartella "output" se non esiste
path = os.path.join("output")
os.makedirs(path, exist_ok=True)

# salva i file nella cartella di output
for ext in ("png", "pdf"):
    fn = os.path.join(path, f"semantic_umap_ilmr.{ext}") #MODIFICA
    plt.savefig(fn, dpi=300)

print("✅ Saved output/semantic_umap.{png,pdf}")

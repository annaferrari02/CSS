#!/usr/bin/env python3
"""
thread_safe_moderation.py

Reads a CSV with a 'text' column, filters out empty rows,
then spins up a pool of threads to call OpenAI's moderation
API. Each thread keeps its own client in thread‐local storage.

Usage:
    python thread_safe_moderation.py /path/to/input.csv /path/to/output_dir
"""

import os
import argparse
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# —— CONFIGURATION —— #
DEFAULT_API_KEY = ""
DEFAULT_MODEL   = "omni-moderation-latest"
DEFAULT_WORKERS = 24

# thread-local storage for each thread’s client
_thread_local = threading.local()

def get_thread_client(model_name: str):
    """
    Create (once per thread) and return an OpenAI client + model name.
    """
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(api_key=DEFAULT_API_KEY)
        _thread_local.model  = model_name
    return _thread_local.client, _thread_local.model

def moderate_text(text: str, model_name: str) -> dict:
    """
    Call the moderation API on 'text' and return
    a simple dict: category_name → score.
    """
    client, model = get_thread_client(model_name)
    resp = client.moderations.create(model=model, input=str(text))
    return resp.results[0].category_scores.to_dict()

def parse_args():
    p = argparse.ArgumentParser(
        description="Thread‐safe parallel moderation with OpenAI’s API"
    )
    p.add_argument(
        "input_csv",
        help="Path to the input CSV (must have a 'text' column)"
    )
    p.add_argument(
        "output_dir",
        help="Directory where the output CSV will be written"
    )
    p.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Moderation model (default: {DEFAULT_MODEL})"
    )
    p.add_argument(
        "--workers", "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of threads (default: {DEFAULT_WORKERS})"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Load & clean
    df = pd.read_csv(args.input_csv)
    if "text" not in df.columns:
        raise ValueError("Input CSV must have a 'text' column.")
    df = df[df["text"].notna() & df["text"].str.strip().astype(bool)].reset_index(drop=True)

    # 2) Spin up ThreadPool
    texts = df["text"].tolist()
    results = [None] * len(texts)

    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        # submit all jobs, keep track of their original indices
        future_to_idx = {
            exe.submit(moderate_text, txt, args.model): i
            for i, txt in enumerate(texts)
        }

        # collect in whichever order they finish, but store by index
        for future in tqdm(as_completed(future_to_idx),
                           total=len(texts),
                           desc="Moderating"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                # on error, store empty dict (or handle as you wish)
                results[idx] = {}
                print(f"[!] Error on row {idx}: {e}")

    # 3) Build scores dataframe
    scores_df = pd.DataFrame(results).fillna(0)
    # sanitize column names: "sexual/minors" → "sexual_minors"
    scores_df.columns = [c.replace("/", "_") for c in scores_df.columns]

    # 4) Concat & write out
    out_df = pd.concat([df, scores_df], axis=1)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, os.path.basename(args.input_csv))
    out_df.to_csv(out_path, index=False)

    print(f"✅ Wrote {len(out_df)} rows with moderation scores to {out_path}")

if __name__ == "__main__":
    main()

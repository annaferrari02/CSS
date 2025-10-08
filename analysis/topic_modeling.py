import os
from pathlib import Path
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Import from shared preprocessing module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocessing.text_cleaner import clean_text, STOPWORDS, LEMMATIZER


def run_topic_modeling(data_path, output_dir=None):
    """
    Run BERTopic analysis on chat data
    
    Args:
        data_path: Path to CSV with 'party' and 'text' columns
        output_dir: Directory to save results (default: ./output)
    """
    # 🔹 Imposta directory di output
    if output_dir is None:
        output_dir = Path("output")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_dir.resolve()}")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, usecols=["party", "text"])
    
    # Clean text
    print("Cleaning text...")
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    
    # Fit BERTopic
    print("Fitting BERTopic model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    topic_model = BERTopic(embedding_model=embedder, verbose=False)
    
    docs = df["clean_text"].tolist()
    topics, _ = topic_model.fit_transform(docs)
    df["topic"] = [int(t) for t in topics]
    
    # Save topic overview
    print("Saving results...")
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(output_dir / "topic_info.csv", index=False)
    
    # Save top terms for each topic
    with open(output_dir / "topic_terms.txt", "w") as f:
        for _, row in topic_info.iterrows():
            t_id = int(row.Topic)
            if t_id < 0:
                continue
            f.write(f"Topic {t_id} ({row.Count} docs):\n")
            for term, weight in topic_model.get_topic(t_id):
                f.write(f"  {term:<15s} {weight:.4f}\n")
            f.write("\n")
    
    # Compute counts & proportions per party/topic
    party_topic = (
        df.groupby(["party", "topic"])
          .size()
          .rename("count")
          .reset_index()
    )
    party_topic["prop"] = party_topic.groupby("party")["count"].transform(lambda x: x / x.sum())
    
    # Extract top-10 topics for USER and Chatbot
    user_top10 = (
        party_topic[party_topic.party == "USER"]
        .nlargest(10, "prop")[["topic", "count", "prop"]]
        .reset_index(drop=True)
    )
    bot_top10 = (
        party_topic[party_topic.party == "Chatbot"]
        .nlargest(10, "prop")[["topic", "count", "prop"]]
        .reset_index(drop=True)
    )
    
    # Save CSVs
    user_top10.to_csv(output_dir / "user_top10_topics.csv", index=False)
    bot_top10.to_csv(output_dir / "bot_top10_topics.csv", index=False)
    
    # Write human-readable summary
    with open(output_dir / "topic_top10_summary.txt", "w") as f:
        f.write("Top 10 Topics by Proportion\n")
        f.write("===========================\n\n")
        f.write("USER:\n")
        for i, row in user_top10.iterrows():
            topic_id = int(row["topic"])
            cnt = int(row["count"])
            prop = row["prop"]
            f.write(f"{i+1:2d}. Topic {topic_id} - count={cnt}, prop={prop:.3f}\n")
        
        f.write("\nChatbot:\n")
        for i, row in bot_top10.iterrows():
            topic_id = int(row["topic"])
            cnt = int(row["count"])
            prop = row["prop"]
            f.write(f"{i+1:2d}. Topic {topic_id} - count={cnt}, prop={prop:.3f}\n")
    
    print("Results saved:")
    for fname in ["topic_info.csv", "topic_terms.txt", "user_top10_topics.csv",
                  "bot_top10_topics.csv", "topic_top10_summary.txt"]:
        print(f"  - {output_dir / fname}")
    
    return topic_model, df


if __name__ == '__main__':
    DATA_PATH = Path("data") / "human-ai-chatlogs.csv"
    
    # Fallback path
    if not DATA_PATH.exists():
        DATA_PATH = Path("C:/Users/anna2/OneDrive/Desktop/CSS/human_ai_chatlogs.csv")
        print(f"Using legacy path: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found at {DATA_PATH}")
        sys.exit(1)
    
    # 🔹 Run analysis con output in ./output/
    topic_model, df = run_topic_modeling(DATA_PATH, output_dir="output")

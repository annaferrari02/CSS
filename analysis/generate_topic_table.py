import os
from pathlib import Path
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from mistralai import Mistral

# Import from shared preprocessing module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from text_cleaner import clean_text, STOPWORDS, LEMMATIZER


def label_topic_with_llm(topic_terms, topic_id, count, api_key=None):
    """
    Usa Mistral AI per generare una label descrittiva per il topic
    
    Args:
        topic_terms: Lista di (term, weight) per il topic
        topic_id: ID del topic
        count: Numero di documenti nel topic
        api_key: Mistral API key (opzionale, può essere in env var)
    
    Returns:
        str: Label descrittiva del topic
    """
    # Prendi i top 10 termini
    top_terms = [term for term, _ in topic_terms[:10]]
    terms_str = ", ".join(top_terms)
    
    prompt = f"""Given these top keywords from a topic in user-chatbot conversations:
{terms_str}

Generate a short, descriptive label (max 6 words) that captures the main theme.
Examples:
- "Affirmations & confirmations"
- "Platform references (Replika, Reddit)"
- "Role-play commands & fantasy RP"
- "Erotic / affectionate descriptions"
- "Greetings & introductions"

Return ONLY the label, nothing else.

Label:"""
    
    try:
        # Inizializza client Mistral
        client = Mistral(api_key=api_key or os.getenv("MISTRAL_API_KEY"))
        
        # Chiama l'API
        response = client.chat.complete(
            model="mistral-large-latest",  # o "mistral-medium" per risparmiare
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3  # Bassa per risposte più consistenti
        )
        
        label = response.choices[0].message.content.strip()
        
        # Rimuovi eventuali quote o caratteri extra
        label = label.strip('"\'')
        
        return label
        
    except Exception as e:
        print(f"⚠️ Error labeling topic {topic_id}: {e}")
        # Fallback a label automatica
        return f"Topic {topic_id}: {', '.join(top_terms[:3])}"

def create_publication_table(data_path, output_dir=None, use_llm_labels=False, mistral_api_key=None):
    """
    Crea una tabella publication-ready come nel paper
    
    Args:
        data_path: Path to CSV with 'party' and 'text' columns
        output_dir: Directory to save results
        use_llm_labels: Se True, usa Mistral AI per generare label descrittive
        mistral_api_key: Mistral API key (opzionale se in env var)
    """
        # Setup output directory
    if output_dir is None:
        output_dir = Path("output")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_dir.resolve()}")

    # Load and clean data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, usecols=["party", "text"])
    
    print("Cleaning text...")
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    
    # Fit BERTopic
    print("Fitting BERTopic model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    topic_model = BERTopic(embedding_model=embedder, verbose=False, nr_topics=20)
    
    docs = df["clean_text"].tolist()
    topics, _ = topic_model.fit_transform(docs)
    df["topic"] = [int(t) for t in topics]
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    # Compute counts per party and topic
    party_topic_counts = (
        df.groupby(["party", "topic"])
          .size()
          .reset_index(name="count")
    )
    
    user_topics = party_topic_counts[party_topic_counts["party"] == "USER"].copy()
    chatbot_topics = party_topic_counts[party_topic_counts["party"] == "Chatbot"].copy()

    user_top8 = user_topics.nlargest(8, "count").reset_index(drop=True)
    chatbot_top8 = chatbot_topics.nlargest(8, "count").reset_index(drop=True)
    
    # Generate labels
    print("Generating topic labels...")
    
    def get_topic_label(topic_id, count):
        if topic_id == -1:
            return "Outliers", count
        
        topic_terms = topic_model.get_topic(topic_id)
        
        if use_llm_labels:
            label = label_topic_with_llm(topic_terms, topic_id, count, mistral_api_key)
        else:
            # Label manuale basato sui top terms
            top_words = [term for term, _ in topic_terms[:5]]
            label = f"Topic {topic_id}: {', '.join(top_words[:3])}"
        
        return label, count
    
    # Create results dataframes
    user_results = []
    for idx, row in user_top8.iterrows():
        label, count = get_topic_label(row["topic"], row["count"])
        user_results.append({
            "Rank": idx + 1,
            "Topic": label,
            "Count": count
        })
    
    chatbot_results = []
    for idx, row in chatbot_top8.iterrows():
        label, count = get_topic_label(row["topic"], row["count"])
        chatbot_results.append({
            "Rank": idx + 1,
            "Topic": label,
            "Count": count
        })
    
    # Save as CSV
    user_df = pd.DataFrame(user_results)
    chatbot_df = pd.DataFrame(chatbot_results)
    
    user_df.to_csv(output_dir / "user_top_topics_table.csv", index=False)
    chatbot_df.to_csv(output_dir / "chatbot_top_topics_table.csv", index=False)
    
    # Create formatted text table (LaTeX/Markdown style)
    with open(output_dir / "publication_table.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TOP TOPICS IN USER-CHATBOT CONVERSATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("USER Topics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Topic':<60} {'Count':<10}\n")
        f.write("-" * 80 + "\n")
        for row in user_results:
            f.write(f"{row['Rank']:<6} {row['Topic']:<60} {row['Count']:<10}\n")
        
        f.write("\n\n")
        f.write("CHATBOT Topics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Topic':<60} {'Count':<10}\n")
        f.write("-" * 80 + "\n")
        for row in chatbot_results:
            f.write(f"{row['Rank']:<6} {row['Topic']:<60} {row['Count']:<10}\n")
    
    # Also save detailed topic terms for manual review
    with open(output_dir / "topic_terms_for_labeling.txt", "w", encoding="utf-8") as f:
        f.write("TOPIC TERMS FOR MANUAL LABELING\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("USER Top 8 Topics:\n\n")
        for idx, row in user_top8.iterrows():
            topic_id = row["topic"]
            if topic_id == -1:
                continue
            f.write(f"Topic {topic_id} (Count: {row['count']}):\n")
            topic_terms = topic_model.get_topic(topic_id)
            for term, weight in topic_terms[:10]:
                f.write(f"  {term:<20} {weight:.4f}\n")
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        f.write("CHATBOT Top 8 Topics:\n\n")
        for idx, row in chatbot_top8.iterrows():
            topic_id = row["topic"]
            if topic_id == -1:
                continue
            f.write(f"Topic {topic_id} (Count: {row['count']}):\n")
            topic_terms = topic_model.get_topic(topic_id)
            for term, weight in topic_terms[:10]:
                f.write(f"  {term:<20} {weight:.4f}\n")
            f.write("\n")
    
    print("\n✅ Results saved:")
    print(f"  - {output_dir / 'user_top_topics_table.csv'}")
    print(f"  - {output_dir / 'chatbot_top_topics_table.csv'}")
    print(f"  - {output_dir / 'publication_table.txt'}")
    print(f"  - {output_dir / 'topic_terms_for_labeling.txt'}")
    
    return topic_model, df, user_df, chatbot_df


if __name__ == '__main__':
    DATA_PATH = Path("data") / "human_ai_chatlogs.csv"
    
    # Fallback path
    if not DATA_PATH.exists():
        DATA_PATH = Path("C:/Users/anna2/OneDrive/Desktop/CSS/data/human_ai_chatlogs.csv")
        print(f"Using legacy path: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found at {DATA_PATH}")
        sys.exit(1)
    
    # Get Mistral API key from environment or hardcode (NOT recommended for production)
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # oppure "your-api-key-here"
    
    # Run analysis
    topic_model, df, user_df, chatbot_df = create_publication_table(
        DATA_PATH, 
        output_dir="output",
        use_llm_labels=True,  # ✅ Attiva Mistral AI
        mistral_api_key=MISTRAL_API_KEY
    )
    
    # Print preview
    print("\n" + "=" * 80)
    print("PREVIEW - USER Topics:")
    print(user_df.to_string(index=False))
    print("\n" + "=" * 80)
    print("PREVIEW - CHATBOT Topics:")
    print(chatbot_df.to_string(index=False))
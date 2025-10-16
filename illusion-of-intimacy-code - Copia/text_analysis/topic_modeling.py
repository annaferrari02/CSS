
import os
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool, cpu_count
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# 1) GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 2) NLTK setup & stopwords
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
    "than","too","very","s","t","can","will","just","don","should","now"
}
stopwords_list = stop_words.union(extra)
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    t = re.sub(r'http\S+|www\S+|t\.co\S+', "", text)
    t = re.sub(r'[^\w\s]', "", t).lower().strip()
    toks = word_tokenize(t)
    return " ".join(
        lemmatizer.lemmatize(w) for w in toks
        if w not in stopwords_list
    )

# 3) Load data & clean in parallel
DATA_PATH = "/home/mhchu/AI-Companion/human-ai/data/data/human_ai_chatlogs.csv"
df = pd.read_csv(DATA_PATH, usecols=["party","text"])

with Pool(cpu_count()) as pool:
    df["clean_text"] = pool.map(clean_text, df["text"].astype(str).tolist())

# 4) Fit BERTopic
embedder    = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
topic_model = BERTopic(embedding_model=embedder, verbose=False)

docs   = df["clean_text"].tolist()
topics, _ = topic_model.fit_transform(docs)
df["topic"] = [int(t) for t in topics]

# 5) Save topic overview
topic_info = topic_model.get_topic_info()
topic_info.to_csv("topic_info.csv", index=False)

# 6) Save top terms for each topic
with open("topic_terms.txt", "w") as f:
    for _, row in topic_info.iterrows():
        t_id = int(row.Topic)
        if t_id < 0:
            continue
        f.write(f"Topic {t_id} ({row.Count} docs):\n")
        for term, weight in topic_model.get_topic(t_id):
            f.write(f"  {term:<15s} {weight:.4f}\n")
        f.write("\n")

# 7) Compute counts & proportions per party/topic
party_topic = (
    df.groupby(["party","topic"])
      .size()
      .rename("count")
      .reset_index()
)
party_topic["prop"] = party_topic.groupby("party")["count"].transform(lambda x: x / x.sum())

# 8) Extract top-10 topics for USER and Chatbot
user_top10 = (
    party_topic[party_topic.party=="USER"]
    .nlargest(10, "prop")[["topic","count","prop"]]
    .reset_index(drop=True)
)
bot_top10 = (
    party_topic[party_topic.party=="Chatbot"]
    .nlargest(10, "prop")[["topic","count","prop"]]
    .reset_index(drop=True)
)

# 9) Save CSVs
user_top10.to_csv("user_top10_topics.csv", index=False)
bot_top10.to_csv("bot_top10_topics.csv", index=False)

# 10) Write human-readable summary
with open("topic_top10_summary.txt", "w") as f:
    f.write("Top 10 Topics by Proportion\n")
    f.write("===========================\n\n")
    f.write("USER:\n")
    for i, row in user_top10.iterrows():
        topic_id = int(row["topic"])
        cnt      = int(row["count"])
        prop     = row["prop"]
        f.write(f"{i+1:2d}. Topic {topic_id} — count={cnt}, prop={prop:.3f}\n")

    f.write("\nChatbot:\n")
    for i, row in bot_top10.iterrows():
        topic_id = int(row["topic"])
        cnt      = int(row["count"])
        prop     = row["prop"]
        f.write(f"{i+1:2d}. Topic {topic_id} — count={cnt}, prop={prop:.3f}\n")

print("✅ user_top10_topics.csv, bot_top10_topics.csv, topic_top10_summary.txt written")

import pandas as pd

# Load data
df = pd.read_csv("/home/mhchu/AI-Companion/human-ai/OpenAI_moderation/results/human_ai_chatlogs.csv")

# Count dialogues with at least one spike above threshold
threshold = 0.5
categories = ["self-harm", "sexual", "violence", "harassment"]

# Calculate total unique dialogues
total_dialogues = df["id"].nunique()

# Count dialogues with spikes
spiked_dialogues = 0
for dlg_id, grp in df.groupby("id"):
    # Fixed this line to properly check if any value exceeds threshold
    if any((grp[cat] > threshold).any() for cat in categories):
        spiked_dialogues += 1

# Calculate percentage
spike_percentage = (spiked_dialogues / total_dialogues) * 100
print(f"Percentage of dialogues with spikes: {spike_percentage:.2f}%")
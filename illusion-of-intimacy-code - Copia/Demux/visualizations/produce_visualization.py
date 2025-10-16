import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("/home/mhchu/AI Companion/human-ai/Demux/emo_results/human_submissions_emo.csv")

df = df[~df["subreddit"].isin(['1694095885.0', '1696550380.0', '1695845938.0'])]

# List of emotion columns
emotion_cols = [
    'anger','disgust','fear','pessimism','sadness',
    'anticipation','surprise','optimism','joy','love','trust'
]

# Convert your pandas DataFrame to a Dask DataFrame with an appropriate number of partitions.
ddf = dd.from_pandas(df, npartitions=16)

# 1. Create a new column 'dominant_emotion' using vectorized idxmax. Dask supports many vectorized operations.
ddf['dominant_emotion'] = ddf[emotion_cols].idxmax(axis=1)

# 2. Group by subreddit and sum emotion columns (this is done in parallel)
emotion_sums = ddf.groupby('subreddit')[emotion_cols].sum()

# Compute the result back into a pandas DataFrame for further processing/plotting
emotion_sums = emotion_sums.compute()

# Convert sums to proportions so each subreddit bar sums to 1
emotion_props = emotion_sums.div(emotion_sums.sum(axis=1), axis=0)

# 3. Plot the stacked bar chart
ax = emotion_props.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Paired')
plt.title('Human Submissions')
plt.xlabel('Subreddit')
plt.ylabel('Proportion of Emotions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


# Save the figure as a PDF file
plt.savefig('/home/mhchu/AI Companion/human-ai/Demux/emo_viz/human_submissions_viz.pdf')

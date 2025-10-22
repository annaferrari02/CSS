# emotion_mirroring_analysis.py
"""
Emotional Mirroring Analysis for Human-AI Conversations
Based on "Illusions of Intimacy" methodology
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR.parent / "data" / "human_ai_chatlogs_ilmr.csv"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Emotions from the paper (8 key emotions)
EMOTION_COLS = ['anger', 'disgust', 'fear', 'sadness', 
                'surprise', 'joy', 'love', 'optimism']

# Color scheme matching the paper
COLORS = {
    'USER': '#FF6B6B',      # Red/pink for users
    'Chatbot': '#4DD0E1'    # Cyan/blue for chatbots
}

# ============================================================================
# STEP 1: EMOTION DETECTION USING ROBERTA (same as paper)
# ============================================================================

def detect_emotions_roberta(texts, batch_size=32):
    """
    Use RoBERTa-based GoEmotions model (same as the paper)
    Returns emotion scores for each text
    """
    print("Loading RoBERTa GoEmotions model (this may take a moment)...")
    
    # Initialize the emotion classifier
    emotion_classifier = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,  # Return all emotion scores
        device=-1    # Use CPU; change to 0 for GPU
    )
    
    print(f"Processing {len(texts)} texts in batches of {batch_size}...")
    
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = emotion_classifier(batch)
        all_results.extend(batch_results)
        
        if (i + batch_size) % 1000 == 0:
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts...")
    
    return all_results

def process_emotion_scores(raw_results, threshold=0.05):
    """
    Process raw emotion scores into dataframe format
    Applies threshold masking as in the paper
    """
    processed = []
    
    for result in raw_results:
        emotion_dict = {}
        for emotion_score in result:
            label = emotion_score['label']
            score = emotion_score['score']
            # Apply threshold mask (paper uses 0.05)
            emotion_dict[label] = score if score >= threshold else 0.0
        processed.append(emotion_dict)
    
    return pd.DataFrame(processed)

# ============================================================================
# STEP 2: RADAR PLOT (Figure 4 from paper)
# ============================================================================

def create_radar_plot(df_emotions, output_path):
    """
    Create radar plot showing dominant emotion distributions
    Matches Figure 4 from the paper
    """
    print("\nCreating radar plot...")
    
    # Get dominant emotion per turn
    df_emotions['dominant'] = df_emotions[EMOTION_COLS].idxmax(axis=1)
    
    # Compute proportions for each party
    proportions = {}
    for party in ['USER', 'Chatbot']:
        sub = df_emotions[df_emotions['party'] == party]
        counts = sub['dominant'].value_counts().reindex(EMOTION_COLS, fill_value=0)
        proportions[party] = counts / counts.sum() if counts.sum() > 0 else counts
    
    # Setup radar plot
    N = len(EMOTION_COLS)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    # Increase font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 13,
        'legend.fontsize': 13
    })
    
    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(polar=True))
    
    # Plot each party
    for party in ['USER', 'Chatbot']:
        vals = proportions[party].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[party], linewidth=4, label=party)
        ax.fill(angles, vals, color=COLORS[party], alpha=0.15)
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(EMOTION_COLS)
    ax.set_yticklabels([])
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path / "emotion_radar.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "emotion_radar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Radar plot saved to {output_path}")
    
    return proportions

# ============================================================================
# STEP 3: TIME SERIES ANALYSIS (Figures 12-13 from paper)
# ============================================================================

def create_emotion_timeseries(df, output_path):
    """
    Create monthly emotion distribution plots
    Matches Figures 12-13 from the paper
    """
    print("\nCreating time series analysis...")
    
    # Check if timestamp exists
    if 'timestamp' not in df.columns:
        print("⚠ No timestamp column found, skipping time series analysis")
        return
    
    # Convert timestamp if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    
    df = df.dropna(subset=['timestamp'])
    df = df.set_index('timestamp')
    
    # Get dominant emotion per turn
    df['dominant'] = df[EMOTION_COLS].idxmax(axis=1)
    
    for party in ['USER', 'Chatbot']:
        df_party = df[df['party'] == party]
        
        # Monthly counts
        counts = (
            df_party['dominant']
            .groupby(pd.Grouper(freq='ME'))
            .value_counts()
            .unstack(fill_value=0)
            .reindex(columns=EMOTION_COLS, fill_value=0)
        )
        
        # Convert to proportions
        props = counts.div(counts.sum(axis=1), axis=0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        props.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            width=0.8,
            color=['#E63946', '#8AC926', '#845EC2', '#264653', 
                   '#FFB30F', '#FFD166', '#FF5D9E', '#06D6A0']
        )
        
        # Format x-axis
        months = props.index.to_period("M").to_timestamp()
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(
            [dt.strftime("%m/%y") for dt in months],
            rotation=45,
            ha="right"
        )
        
        ax.set_xlabel("")
        ax.set_ylabel(f"Proportion of Emotions ({party})")
        ax.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
        
        plt.tight_layout()
        plt.savefig(output_path / f"emotion_timeseries_{party.lower()}.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(output_path / f"emotion_timeseries_{party.lower()}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Time series plots saved to {output_path}")

# ============================================================================
# STEP 4: STATISTICAL ANALYSIS (Table 2 from paper)
# ============================================================================

def compute_mirroring_statistics(df_emotions, output_path):
    """
    Compute statistical tests for emotional mirroring
    Matches Table 2 analysis from the paper
    """
    from scipy.stats import ttest_rel, wilcoxon
    
    print("\nComputing mirroring statistics...")
    
    # Aggregate by dialogue and party
    agg = (
        df_emotions
        .groupby(['conversation_id', 'party'])[EMOTION_COLS]
        .mean()
        .reset_index()
    )
    
    # Split by party
    user = agg[agg['party'] == 'USER'].set_index('conversation_id')[EMOTION_COLS]
    bot = agg[agg['party'] == 'Chatbot'].set_index('conversation_id')[EMOTION_COLS]
    
    # Merge on common dialogues
    merged = user.join(bot, how='inner', lsuffix='_user', rsuffix='_bot')
    
    results = []
    
    for emo in EMOTION_COLS:
        u = merged[f'{emo}_user']
        b = merged[f'{emo}_bot']
        
        # Paired t-test
        if len(u) >= 3:
            try:
                stat, p = ttest_rel(b, u, nan_policy='omit')
                
                mean_u = u.mean()
                mean_b = b.mean()
                diff = mean_b - mean_u
                
                if p < 0.05:
                    direction = "Higher" if diff > 0 else "Lower"
                else:
                    direction = "No difference"
                
                results.append({
                    'Emotion': emo,
                    'User_Mean': mean_u,
                    'Chatbot_Mean': mean_b,
                    'Difference': diff,
                    'p_value': p,
                    'Conclusion': direction,
                    'N_dialogues': len(u)
                })
            except:
                pass
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / "mirroring_statistics.csv", index=False)
    
    # Print summary
    print("\n=== Mirroring Statistics (Dialogue-Level) ===")
    print(results_df.to_string(index=False))
    print(f"\n✓ Statistics saved to {output_path / 'mirroring_statistics.csv'}")
    
    return results_df

# ============================================================================
# STEP 5: TURN-LEVEL COSINE SIMILARITY
# ============================================================================

def compute_turn_level_similarity(df_emotions, output_path):
    """
    Compute cosine similarity between user and chatbot emotion vectors
    Mentioned in the paper's turn-level analysis
    """
    from scipy.spatial.distance import cosine
    
    print("\nComputing turn-level similarity...")
    
    # Build macro-turns (consecutive messages from same speaker)
    df_sorted = df_emotions.sort_values(['conversation_id', 'turn'])
    df_sorted['run'] = (
        df_sorted['party'] != df_sorted.groupby('conversation_id')['party'].shift()
    ).cumsum()
    
    macro = (
        df_sorted.groupby(['conversation_id', 'run', 'party'])[EMOTION_COLS]
        .mean()
        .reset_index()
    )
    macro['macro_turn'] = macro.groupby('conversation_id').cumcount() + 1
    
    # Pair user -> chatbot turns
    user = macro[macro['party'] == 'USER'].copy()
    user['next_turn'] = user['macro_turn'] + 1
    
    bot = macro[macro['party'] == 'Chatbot']
    
    pairs = pd.merge(
        user, bot,
        left_on=['conversation_id', 'next_turn'],
        right_on=['conversation_id', 'macro_turn'],
        suffixes=('_u', '_b')
    )
    
    # Compute cosine similarity for each pair
    similarities = []
    for _, row in pairs.iterrows():
        u_vec = row[[e+'_u' for e in EMOTION_COLS]].fillna(0).values
        b_vec = row[[e+'_b' for e in EMOTION_COLS]].fillna(0).values
        
        sim = 1 - cosine(u_vec, b_vec)
        similarities.append(sim)
    
    mean_sim = np.mean(similarities)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(similarities, bins=50, color='#4DD0E1', alpha=0.7, edgecolor='black')
    ax.axvline(mean_sim, color='#FF6B6B', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_sim:.3f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('Turn-Level Emotion Vector Similarity (User → Chatbot)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "cosine_similarity_distribution.pdf", 
               dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "cosine_similarity_distribution.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mean cosine similarity: {mean_sim:.4f}")
    print(f"✓ Similarity plot saved to {output_path}")
    
    return mean_sim

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("EMOTIONAL MIRRORING ANALYSIS")
    print("Based on 'Illusions of Intimacy' methodology")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {len(df)} dialogue turns")
    
    # Check required columns
    required = ['conversation_id', 'party', 'text']
    if 'turn' not in df.columns:
        # Create turn numbers if not present
        df['turn'] = df.groupby('conversation_id').cumcount() + 1
    
    # Remove empty texts
    df = df[df['text'].notna() & (df['text'] != '')]
    print(f"✓ {len(df)} turns after removing empty texts")
    
    # Detect emotions using RoBERTa (same as paper)
    print("\n" + "="*70)
    print("STEP 1: EMOTION DETECTION")
    print("="*70)
    
    texts = df['text'].tolist()
    raw_emotions = detect_emotions_roberta(texts, batch_size=32)
    emotion_scores = process_emotion_scores(raw_emotions, threshold=0.05)
    
    # Combine with original data
    df_emotions = pd.concat([df.reset_index(drop=True), emotion_scores], axis=1)
    
    # Save processed data
    df_emotions.to_csv(OUTPUT_DIR / "dialogue_emotions_full.csv", index=False)
    print(f"✓ Saved emotion scores to {OUTPUT_DIR / 'dialogue_emotions_full.csv'}")
    
    # Create visualizations and analyses
    print("\n" + "="*70)
    print("STEP 2: RADAR PLOT (Figure 4)")
    print("="*70)
    proportions = create_radar_plot(df_emotions, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("STEP 3: TIME SERIES ANALYSIS (Figures 12-13)")
    print("="*70)
    create_emotion_timeseries(df_emotions, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("STEP 4: STATISTICAL ANALYSIS (Table 2)")
    print("="*70)
    stats = compute_mirroring_statistics(df_emotions, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("STEP 5: TURN-LEVEL SIMILARITY")
    print("="*70)
    sim = compute_turn_level_similarity(df_emotions, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - dialogue_emotions_full.csv (complete emotion scores)")
    print("  - emotion_radar.pdf/png (Figure 4 style)")
    print("  - emotion_timeseries_*.pdf/png (Figures 12-13 style)")
    print("  - mirroring_statistics.csv (Table 2 style)")
    print("  - cosine_similarity_distribution.pdf/png")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
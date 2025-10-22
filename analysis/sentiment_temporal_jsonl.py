"""
SENTIMENT ANALYSIS TEMPORALE PER REDDIT DATA
Legge file JSONL e analizza sentiment con distanze temporali

INSTALLAZIONE:
pip install pandas numpy matplotlib seaborn scipy textblob vaderSentiment
python -m textblob.download_corpora

USO:
python sentiment_temporal_jsonl.py
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Setup
vader = SentimentIntensityAnalyzer()
sns.set_style("whitegrid")

# ===== CONFIGURAZIONE PERCORSI =====
# Directory base del progetto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / 'output' / 'sentiment_analysis'

# Crea directory di output se non esiste
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File da analizzare (nella directory reddit_data)
FILES_TO_ANALYZE = [DATA_DIR / 'r_MyBoyfriendIsAI_comments.jsonl'
]

MESI_DA_CONFRONTARE = [1, 3, 6]  # Confronta mese 1, 3 e 6


# ===== CARICAMENTO DATI JSONL =====
def carica_jsonl(filepath, max_rows=None):
    """
    Carica file JSONL (un JSON per riga)
    max_rows: limita numero righe (utile per test con file grandi)
    """
    print(f"📂 Caricamento {filepath.name}...")
    
    if not filepath.exists():
        print(f"   ❌ File non trovato: {filepath}")
        return None
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not data:
        print(f"   ❌ Nessun dato caricato")
        return None
    
    df = pd.DataFrame(data)
    print(f"   ✅ Caricati {len(df)} commenti")
    
    return df


def normalizza_dataframe(df):
    """
    Adatta il DataFrame a diverse strutture possibili
    """
    # Possibili nomi colonne per il testo
    text_columns = ['body', 'text', 'selftext', 'comment', 'content']
    text_col = None
    for col in text_columns:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        print("❌ Colonna testo non trovata!")
        print(f"Colonne disponibili: {list(df.columns)}")
        return None
    
    # Rinomina colonna testo
    df = df.rename(columns={text_col: 'testo'})
    
    # Possibili nomi per timestamp
    time_columns = ['created_utc', 'timestamp', 'created', 'date']
    time_col = None
    for col in time_columns:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        print("❌ Colonna timestamp non trovata!")
        return None
    
    # Converti timestamp
    if time_col == 'created_utc':
        df['timestamp'] = pd.to_datetime(df[time_col], unit='s')
    else:
        df['timestamp'] = pd.to_datetime(df[time_col])
    
    # Possibili nomi per autore
    author_columns = ['author', 'user', 'username']
    author_col = None
    for col in author_columns:
        if col in df.columns:
            author_col = col
            break
    
    if author_col:
        df = df.rename(columns={author_col: 'autore'})
    else:
        df['autore'] = 'unknown'
    
    # Possibili nomi per score
    if 'score' not in df.columns:
        if 'ups' in df.columns:
            df['score'] = df['ups']
        else:
            df['score'] = 0
    
    # Filtra commenti vuoti o deleted
    df = df[df['testo'].notna()].copy()
    df = df[~df['testo'].isin(['[deleted]', '[removed]', ''])].copy()
    
    # Aggiungi subreddit se presente
    if 'subreddit' not in df.columns:
        df['subreddit'] = 'unknown'
    
    print(f"   ✅ DataFrame normalizzato: {len(df)} commenti validi")
    print(f"   📅 Periodo: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    return df[['testo', 'timestamp', 'autore', 'score', 'subreddit']]


def carica_multipli_file(filepaths, max_rows_per_file=None):
    """Carica e combina multipli file JSONL"""
    print("\n" + "="*70)
    print("CARICAMENTO DATI")
    print("="*70 + "\n")
    
    all_dfs = []
    for filepath in filepaths:
        df = carica_jsonl(filepath, max_rows=max_rows_per_file)
        if df is not None:
            df = normalizza_dataframe(df)
            if df is not None:
                # Aggiungi info file come subreddit se mancante
                if df['subreddit'].iloc[0] == 'unknown':
                    subreddit_name = filepath.stem.replace('_comments', '').replace('_posts', '').replace('r_', '')
                    df['subreddit'] = subreddit_name
                all_dfs.append(df)
    
    if not all_dfs:
        print("❌ Nessun file caricato con successo!")
        return None
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n✅ TOTALE: {len(combined)} commenti da {combined['subreddit'].nunique()} subreddit")
    print(f"📅 Range temporale: {combined['timestamp'].min().date()} → {combined['timestamp'].max().date()}")
    
    return combined


# ===== SENTIMENT ANALYSIS =====
def aggiungi_sentiment(df):
    """Aggiunge metriche sentiment"""
    print("\n🧠 Calcolo sentiment...")
    
    # VADER
    vader_scores = df['testo'].apply(lambda x: vader.polarity_scores(str(x)))
    df['sentiment_compound'] = vader_scores.apply(lambda x: x['compound'])
    df['sentiment_positive'] = vader_scores.apply(lambda x: x['pos'])
    df['sentiment_negative'] = vader_scores.apply(lambda x: x['neg'])
    
    # TextBlob
    textblob_scores = df['testo'].apply(lambda x: TextBlob(str(x)).sentiment)
    df['sentiment_polarity'] = textblob_scores.apply(lambda x: x.polarity)
    df['sentiment_subjectivity'] = textblob_scores.apply(lambda x: x.subjectivity)
    
    # Categorizza
    df['sentiment_category'] = pd.cut(
        df['sentiment_compound'],
        bins=[-1, -0.05, 0.05, 1],
        labels=['negative', 'neutral', 'positive']
    )
    
    print("   ✅ Sentiment calcolato")
    return df


# ===== ANALISI TEMPORALE =====
def confronto_periodi(df, mesi=[1, 3, 6]):
    """Confronta sentiment a distanze temporali"""
    print("\n📊 CONFRONTO TRA PERIODI TEMPORALI\n")
    
    start_date = df['timestamp'].min()
    periodi = {}
    
    for mese in mesi:
        end_date = start_date + timedelta(days=30*mese)
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] < end_date)
        periodi[f'Mese_{mese}'] = df[mask]
    
    # Statistiche per periodo
    risultati = []
    for nome, periodo_df in periodi.items():
        if len(periodo_df) > 0:
            risultati.append({
                'periodo': nome,
                'n_commenti': len(periodo_df),
                'sentiment_mean': periodo_df['sentiment_compound'].mean(),
                'sentiment_std': periodo_df['sentiment_compound'].std(),
                'positive_pct': (periodo_df['sentiment_category'] == 'positive').sum() / len(periodo_df) * 100,
                'negative_pct': (periodo_df['sentiment_category'] == 'negative').sum() / len(periodo_df) * 100,
                'subjectivity': periodo_df['sentiment_subjectivity'].mean()
            })
    
    risultati_df = pd.DataFrame(risultati)
    print(risultati_df.to_string(index=False))
    
    # Test statistico
    print("\n📈 TEST STATISTICI (t-test tra periodi consecutivi):\n")
    periodi_list = list(periodi.values())
    for i in range(len(periodi_list)-1):
        p1 = periodi_list[i]['sentiment_compound'].dropna()
        p2 = periodi_list[i+1]['sentiment_compound'].dropna()
        
        if len(p1) > 30 and len(p2) > 30:  # Minimo campione
            t_stat, p_value = stats.ttest_ind(p1, p2)
            sig = "✓ SIGNIFICATIVO" if p_value < 0.05 else "✗ Non significativo"
            print(f"Mese {mesi[i]} vs Mese {mesi[i+1]}: p={p_value:.4f} {sig}")
    
    return risultati_df, periodi


# ===== VISUALIZZAZIONI =====
def plot_evoluzione_temporale(df):
    """Grafico evoluzione sentiment"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Sentiment giornaliero
    daily = df.groupby(df['timestamp'].dt.date)['sentiment_compound'].mean()
    rolling_7d = daily.rolling(window=7, min_periods=1).mean()
    
    axes[0].plot(daily.index, daily.values, alpha=0.3, color='steelblue', label='Giornaliero')
    axes[0].plot(rolling_7d.index, rolling_7d.values, linewidth=2, color='darkblue', label='Media mobile 7gg')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('Evoluzione Sentiment nel Tempo', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Sentiment Compound')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 2. Distribuzione per mese
    df['mese'] = df['timestamp'].dt.to_period('M')
    monthly_dist = df.groupby(['mese', 'sentiment_category']).size().unstack(fill_value=0)
    monthly_dist_pct = monthly_dist.div(monthly_dist.sum(axis=1), axis=0) * 100
    
    monthly_dist_pct.plot(kind='bar', stacked=True, ax=axes[1],
                          color=['#FF6B6B', '#FFD93D', '#6BCB77'])
    axes[1].set_title('Distribuzione Sentiment per Mese', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Percentuale')
    axes[1].set_xlabel('Mese')
    axes[1].legend(title='Sentiment')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Volatilità
    monthly_vol = df.groupby('mese')['sentiment_compound'].std()
    axes[2].bar(range(len(monthly_vol)), monthly_vol.values, color='coral')
    axes[2].set_title('Volatilità Sentiment per Mese', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Deviazione Standard')
    axes[2].set_xlabel('Mese')
    axes[2].set_xticks(range(len(monthly_vol)))
    axes[2].set_xticklabels([str(m) for m in monthly_vol.index], rotation=45, ha='right')
    axes[2].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'sentiment_evoluzione_temporale.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Salvato: {output_path}")
    plt.close()


def plot_confronto_periodi(risultati_df):
    """Confronto visivo tra periodi"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sentiment medio
    axes[0].bar(risultati_df['periodo'], risultati_df['sentiment_mean'],
               color='steelblue', edgecolor='black')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('Sentiment Medio per Periodo', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Sentiment Compound')
    axes[0].grid(alpha=0.3, axis='y')
    
    # Composizione
    x = range(len(risultati_df))
    width = 0.35
    axes[1].bar([i - width/2 for i in x], risultati_df['positive_pct'],
               width, label='Positive', color='#6BCB77')
    axes[1].bar([i + width/2 for i in x], risultati_df['negative_pct'],
               width, label='Negative', color='#FF6B6B')
    axes[1].set_title('% Commenti Positivi vs Negativi', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Percentuale')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(risultati_df['periodo'])
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'sentiment_confronto_periodi.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Salvato: {output_path}")
    plt.close()


def plot_per_subreddit(df):
    """Confronto tra subreddit"""
    if df['subreddit'].nunique() <= 1:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sentiment medio per subreddit
    sub_sentiment = df.groupby('subreddit')['sentiment_compound'].mean().sort_values()
    axes[0].barh(sub_sentiment.index, sub_sentiment.values, color='teal')
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('Sentiment Medio per Subreddit', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sentiment Compound')
    axes[0].grid(alpha=0.3, axis='x')
    
    # Distribuzione categorie per subreddit
    sub_dist = df.groupby(['subreddit', 'sentiment_category']).size().unstack(fill_value=0)
    sub_dist_pct = sub_dist.div(sub_dist.sum(axis=1), axis=0) * 100
    sub_dist_pct.plot(kind='bar', stacked=True, ax=axes[1],
                     color=['#FF6B6B', '#FFD93D', '#6BCB77'])
    axes[1].set_title('Distribuzione Sentiment per Subreddit', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Percentuale')
    axes[1].set_xlabel('Subreddit')
    axes[1].legend(title='Sentiment')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'sentiment_per_subreddit.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Salvato: {output_path}")
    plt.close()


# ===== PIPELINE COMPLETA =====
def analisi_completa(files, mesi_confronto=[1, 3, 6], max_rows_test=None):
    """
    Pipeline completa di analisi
    
    files: lista percorsi file JSONL
    mesi_confronto: lista mesi da confrontare
    max_rows_test: limita righe per test (None = tutti i dati)
    """
    print("\n" + "="*70)
    print("  SENTIMENT ANALYSIS TEMPORALE - REDDIT AI SUBREDDITS")
    print("="*70)
    
    # 1. Carica dati
    df = carica_multipli_file(files, max_rows_per_file=max_rows_test)
    if df is None:
        return None, None
    
    # 2. Calcola sentiment
    df = aggiungi_sentiment(df)
    
    # 3. Analisi temporale
    risultati_periodi, periodi_dict = confronto_periodi(df, mesi=mesi_confronto)
    
    # 4. Visualizzazioni
    print("\n📊 Generazione visualizzazioni...")
    plot_evoluzione_temporale(df)
    plot_confronto_periodi(risultati_periodi)
    plot_per_subreddit(df)
    
    # 5. Salva risultati
    print("\n💾 Salvataggio risultati...")
    csv_path1 = OUTPUT_DIR / 'sentiment_temporale_completo.csv'
    csv_path2 = OUTPUT_DIR / 'confronto_periodi.csv'
    df.to_csv(csv_path1, index=False)
    risultati_periodi.to_csv(csv_path2, index=False)
    print(f"   ✅ Salvato: {csv_path1}")
    print(f"   ✅ Salvato: {csv_path2}")
    
    # 6. Report finale
    print("\n" + "="*70)
    print("  REPORT FINALE")
    print("="*70)
    
    print(f"\n📊 Commenti analizzati: {len(df):,}")
    print(f"📅 Periodo: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"🏷️ Subreddit: {', '.join(df['subreddit'].unique())}")
    
    print(f"\n📈 Sentiment medio globale: {df['sentiment_compound'].mean():.3f}")
    
    # Calcola trend
    if len(df) > 2000:
        delta_globale = df.iloc[-1000:]['sentiment_compound'].mean() - df.iloc[:1000]['sentiment_compound'].mean()
        trend = "↗️ AUMENTO" if delta_globale > 0 else "↘️ CALO"
        print(f"📊 Trend sentiment: {trend} ({delta_globale:+.3f})")
    
    dist = df['sentiment_category'].value_counts()
    print(f"\n📊 Distribuzione:")
    for cat in ['positive', 'neutral', 'negative']:
        if cat in dist.index:
            pct = (dist[cat] / len(df)) * 100
            print(f"   {cat.capitalize()}: {pct:.1f}%")
    
    print("\n✅ Analisi completata!\n")
    print("File generati:")
    print(f"   - {csv_path1.name}")
    print(f"   - {csv_path2.name}")
    print("   - sentiment_evoluzione_temporale.png")
    print("   - sentiment_confronto_periodi.png")
    print("   - sentiment_per_subreddit.png")
    print(f"\nTutti i file salvati in: {OUTPUT_DIR}")
    
    return df, risultati_periodi


# ===== ESECUZIONE =====
if __name__ == "__main__":
    # Verifica che la directory dati esista
    if not DATA_DIR.exists():
        print(f"\n❌ ERRORE: Directory {DATA_DIR} non trovata!")
        print(f"Crea la directory e inserisci i file JSONL dei subreddit.")
        exit(1)
    
    # Verifica che esistano i file
    files_esistenti = [f for f in FILES_TO_ANALYZE if f.exists()]
    if not files_esistenti:
        print(f"\n❌ ERRORE: Nessun file trovato in {DATA_DIR}")
        print("File cercati:")
        for f in FILES_TO_ANALYZE:
            print(f"  - {f.name}")
        exit(1)
    
    # TEST MODE: carica solo 10000 righe per file per test veloce
    # Per analisi completa, usa max_rows_test=None
    
    print("\n⚠️ MODALITÀ TEST: Caricamento solo prime 10000 righe per file")
    print("Per analisi completa, modifica max_rows_test=None nella funzione sotto\n")
    
    df, periodi = analisi_completa(
        files_esistenti,
        mesi_confronto=MESI_DA_CONFRONTARE,
        max_rows_test=10000  # Cambia a None per analisi completa
    )
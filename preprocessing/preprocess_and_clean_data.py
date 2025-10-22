#!/usr/bin/env python3
"""
preprocess_and_clean_data.py

Questo script esegue due passaggi fondamentali di pulizia:
1.  Filtra i testi basandosi sulla "familiarità" delle parole, ovvero la percentuale
    di parole riconosciute da un dizionario inglese o italiano. Questo serve a
    eliminare i risultati di bassa qualità dell'OCR.
2.  Assicura che la colonna 'turn' esista, dato che è fondamentale per le analisi successive.

Crea un nuovo file CSV pulito che verrà usato da tutti gli altri script di analisi.
"""

import pandas as pd
import enchant
import re
import os
from tqdm import tqdm

# --- CONFIGURAZIONE ---
INPUT_CSV = r"D:\VS CODE DIRECTORY\PYTHON\CSS\moderation\output\perspective_moderated_human_ai_chatlogs.csv"
OUTPUT_CSV = r"D:\VS CODE DIRECTORY\PYTHON\CSS\moderation\output\chatlogs_cleaned.csv"

# Soglia di familiarità: se meno del 40% delle parole sono note, scartiamo il testo.
# Puoi aggiustare questo valore dopo qualche test manuale.
FAMILIARITY_THRESHOLD = 0.40

# Lunghezza minima del testo (in parole) per essere considerato valido.
MIN_WORD_COUNT = 1

# --- FUNZIONE DI CONTROLLO QUALITÀ ---

def calculate_word_familiarity(text: str, dict_en, dict_it) -> float:
    """
    Calcola la percentuale di parole in un testo che sono riconosciute
    da un dizionario inglese O italiano.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    # Pulisce e divide il testo in parole (token)
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) == 0:
        return 0.0

    known_words = 0
    for word in words:
        # Una parola è "conosciuta" se esiste in almeno uno dei due dizionari
        if dict_en.check(word) or dict_it.check(word):
            known_words += 1
            
    return known_words / len(words)

# --- SCRIPT PRINCIPALE ---

def main():
    print("Avvio del processo di pulizia e preprocessing...")

    # 1. Carica i dati
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"ERRORE: File non trovato a '{INPUT_CSV}'")
        return

    print(f"Caricate {len(df)} righe dal file di input.")

    # 2. Inizializza i dizionari
    try:
        dict_en = enchant.Dict("en_US")
        dict_it = enchant.Dict("it_IT")
    except enchant.errors.DictNotFoundError as e:
        print(f"ERRORE: Dizionario non trovato. {e}")
        print("Assicurati di aver installato i dizionari (es. hunspell-en-us, hunspell-it).")
        return

    # 3. Calcola il punteggio di familiarità per ogni riga
    # tqdm aggiunge una barra di progresso, utile per file grandi
    tqdm.pandas(desc="Analisi qualità OCR")
    df['ocr_quality_score'] = df['text'].progress_apply(
        lambda txt: calculate_word_familiarity(txt, dict_en, dict_it)
    )

    # 4. Filtra il DataFrame
    initial_rows = len(df)
    
    # Filtro 1: Qualità OCR
    df_cleaned = df[df['ocr_quality_score'] >= FAMILIARITY_THRESHOLD]
    
    # Filtro 2: Lunghezza minima
    # Calcoliamo il numero di parole e filtriamo
    df_cleaned['word_count'] = df_cleaned['text'].str.split().str.len().fillna(0)
    df_cleaned = df_cleaned[df_cleaned['word_count'] >= MIN_WORD_COUNT]

    final_rows = len(df_cleaned)
    removed_rows = initial_rows - final_rows
    
    print(f"\nFiltraggio completato:")
    print(f" - Righe iniziali: {initial_rows}")
    print(f" - Righe rimosse per bassa qualità OCR o lunghezza insufficiente: {removed_rows}")
    print(f" - Righe rimanenti: {final_rows} ({final_rows / initial_rows:.2%})")

    # 5. Assicura l'esistenza della colonna 'turn'
    if 'turn' not in df_cleaned.columns:
        print("La colonna 'turn' non è presente. Verrà generata ora.")
        # Ordina per sicurezza prima di assegnare il turno, anche se groupby dovrebbe gestire l'ordine
        df_cleaned = df_cleaned.sort_values(by=['conversation_id', 'party']).reset_index(drop=True)
        df_cleaned['turn'] = df_cleaned.groupby('conversation_id').cumcount() + 1

    # 6. Salva il file pulito
    # Rimuoviamo le colonne di servizio che non servono più
    df_cleaned = df_cleaned.drop(columns=['ocr_quality_score', 'word_count'], errors='ignore')
    df_cleaned.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ File pulito salvato con successo in:\n{OUTPUT_CSV}")

if __name__ == "__main__":
    main()
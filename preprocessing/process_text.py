import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
INPUT_FILE = DATA_DIR / 'extracted_chat.txt'
OUTPUT_FILE = DATA_DIR / 'human_ai_chatlogs.csv'


def parse_chat_file_optimized(file_path):
    """
    Parse TXT file - VERSIONE OTTIMIZZATA
    Evita loop annidati e usa list comprehension
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    
    sections = content.split('=' * 80)
    
    # Raccolta dati ottimizzata
    all_messages = []
    conversation_counter = 1
    
    for section in sections:
        if not section.strip():
            continue
        
        lines = section.strip().split('\n')
        current_conv_id = f"conv_{conversation_counter:03d}"
        section_messages = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(r'^(USER|Chatbot):\s*(.*)$', line)
            if match:
                party, text = match.groups()
                text = text.strip()
                
                if text:
                    section_messages.append({
                        'conversation_id': current_conv_id,
                        'party': party,
                        'text': text
                    })
        
        if section_messages:
            all_messages.extend(section_messages)
            conversation_counter += 1
    
    return pd.DataFrame(all_messages)


def clean_text_vectorized(text):
    """Pulisce il testo - versione singola per vectorization"""
    text = ' '.join(text.split())
    
    replacements = {
        ' .': '.', ' ,': ',', ' !': '!', 
        ' ?': '?', ' ;': ';', ' :': ':'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def merge_fragmented_messages_optimized(df):
    """
    VERSIONE ULTRA-OTTIMIZZATA: O(n) invece di O(n²)
    
    Usa groupby + apply invece di loop annidati
    Evita iterrows() completamente
    """
    if df.empty:
        return df
    
    def merge_conversation(conv_group):
        """Merge messaggi consecutivi dello stesso speaker"""
        merged = []
        
        # Converti in lista una volta sola
        messages = conv_group[['party', 'text']].values.tolist()
        
        if not messages:
            return pd.DataFrame()
        
        current_party = messages[0][0]
        current_texts = [messages[0][1]]
        
        for party, text in messages[1:]:
            if party == current_party:
                current_texts.append(text)
            else:
                # Salva messaggio corrente
                merged.append({
                    'party': current_party,
                    'text': clean_text_vectorized(' '.join(current_texts))
                })
                
                # Inizia nuovo messaggio
                current_party = party
                current_texts = [text]
        
        # Aggiungi ultimo messaggio
        merged.append({
            'party': current_party,
            'text': clean_text_vectorized(' '.join(current_texts))
        })
        
        return pd.DataFrame(merged)
    
    # GroupBy + Apply = O(n) invece di loop annidato
    merged_df = df.groupby('conversation_id', sort=False).apply(
        merge_conversation
    ).reset_index(level=1, drop=True).reset_index()
    
    return merged_df


def merge_fragmented_messages_fastest(df):
    """
    VERSIONE PIÙ VELOCE: usa shift() e cumsum() per identificare gruppi
    Approccio completamente vettorizzato
    """
    if df.empty:
        return df
    
    # Identifica quando cambia speaker o conversazione
    df = df.copy()
    df['conv_party'] = df['conversation_id'] + '_' + df['party']
    df['is_new_group'] = (df['conv_party'] != df['conv_party'].shift(1)).astype(int)
    df['group_id'] = df['is_new_group'].cumsum()
    
    # Aggrega per gruppo
    merged = df.groupby('group_id').agg({
        'conversation_id': 'first',
        'party': 'first',
        'text': lambda x: ' '.join(x)  # Unisci testi
    }).reset_index(drop=True)
    
    # Pulisci testi (applica a colonna intera)
    merged['text'] = merged['text'].apply(clean_text_vectorized)
    
    return merged[['conversation_id', 'party', 'text']]


def display_stats(df):
    """Mostra statistiche dettagliate"""
    print(f"\n📊 Statistics:")
    print(f"  Total messages: {len(df)}")
    print(f"  Conversations: {df['conversation_id'].nunique()}")
    print(f"  USER messages: {len(df[df['party'] == 'USER'])}")
    print(f"  Chatbot messages: {len(df[df['party'] == 'Chatbot'])}")
    
    # Statistiche aggiuntive
    avg_msg_length = df['text'].str.len().mean()
    print(f"  Average message length: {avg_msg_length:.1f} characters")
    
    # Distribuzione messaggi per conversazione
    msgs_per_conv = df.groupby('conversation_id').size()
    print(f"  Avg messages per conversation: {msgs_per_conv.mean():.1f}")
    print(f"  Min/Max messages: {msgs_per_conv.min()}/{msgs_per_conv.max()}")


def main():
    print("="*60)
    print("PROCESSING EXTRACTED CHAT TEXT (OPTIMIZED)")
    print("="*60)
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("="*60)
    
    if not INPUT_FILE.exists():
        print(f"\n❌ Error: Input file not found!")
        print(f"Expected: {INPUT_FILE}")
        print("\nRun extract_text.py first to generate the input file.")
        return
    
    print(f"\n🔄 Processing {INPUT_FILE.name}...")
    
    try:
        # Parsing ottimizzato
        df = parse_chat_file_optimized(INPUT_FILE)
        
        if df.empty:
            print("❌ No valid chat conversations found!")
            return
        
        print(f"✓ Found {len(df)} raw messages in {df['conversation_id'].nunique()} conversations")
        
        # Usa la versione più veloce (vettorizzata)
        print("🔄 Merging fragmented messages...")
        df_merged = merge_fragmented_messages_fastest(df)
        
        print(f"✓ After merging: {len(df_merged)} messages")
        
        # Salva risultato
        df_merged.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        
        print(f"\n✅ Successfully created {OUTPUT_FILE.name}")
        
        display_stats(df_merged)
        
        print("\n📄 Preview (first 10 rows):")
        print(df_merged.head(10).to_string(index=False, max_colwidth=60))
        
        # Avviso se ci sono conversazioni molto corte
        short_convs = df_merged.groupby('conversation_id').size()
        if (short_convs == 1).any():
            num_short = (short_convs == 1).sum()
            print(f"\n⚠️  Warning: {num_short} conversations have only 1 message")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
import pandas as pd
from pathlib import Path
import glob

def merge_annotations_from_folders(
    annotators_dir='./validation_sample/annotators',
    output_path='./validation_sample/gold_standard_annotations.csv'
):
    """
    Cerca automaticamente i file completati nelle cartelle annotator_X/
    
    Args:
        annotators_dir: directory con le cartelle annotator_1, annotator_2, etc.
        output_path: dove salvare il risultato combinato
    """
    
    print("="*70)
    print("MERGING ANNOTATIONS FROM GIT REPO")
    print("="*70)
    
    annotators_path = Path(annotators_dir)
    
    if not annotators_path.exists():
        print(f"\n❌ ERROR: Directory not found: {annotators_path}")
        return None
    
    # Cerca tutti i file Excel nelle sottocartelle
    print(f"\nSearching for annotation files in: {annotators_path}")
    
    # Pattern per trovare i file completati
    patterns = [
        '**/annotator_*_COMPLETED.xlsx',  # File rinominati con _COMPLETED
        '**/annotator_*_template.xlsx',    # File originali (se non rinominati)
    ]
    
    all_files = []
    for pattern in patterns:
        found = list(annotators_path.glob(pattern))
        all_files.extend(found)
    
    # Rimuovi duplicati (caso in cui ci siano sia template che COMPLETED)
    unique_files = {}
    for f in all_files:
        # Usa la cartella parent come chiave (annotator_1, annotator_2, etc.)
        annotator = f.parent.name
        if annotator not in unique_files:
            unique_files[annotator] = f
        elif 'COMPLETED' in f.name:
            # Preferisci il file _COMPLETED se esiste
            unique_files[annotator] = f
    
    annotator_files = list(unique_files.values())
    
    print(f"\nFound {len(annotator_files)} annotation files:")
    for f in sorted(annotator_files):
        print(f"  - {f.parent.name}/{f.name}")
    
    if len(annotator_files) == 0:
        print("\n❌ No annotation files found!")
        print("Expected files like:")
        print("  - annotators/annotator_1/annotator_1_template.xlsx")
        print("  - annotators/annotator_1/annotator_1_COMPLETED.xlsx")
        return None
    
    # Carica e combina
    all_annotations = []
    
    for filepath in sorted(annotator_files):
        print(f"\n{'─'*70}")
        print(f"Loading: {filepath.parent.name}/{filepath.name}")
        
        try:
            df = pd.read_excel(filepath)
            
            # Verifica colonne essenziali
            required_cols = ['screenshot_id', 'dialogue_id', 'turn_id', 'speaker', 'text', 'annotator_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  ❌ Missing columns: {missing_cols}")
                continue
            
            # Filtra solo righe completate
            completed = df[df['text'].notna() & (df['text'] != '')].copy()
            
            if len(completed) == 0:
                print(f"  ⚠️  WARNING: No completed annotations found!")
                print(f"     File might be empty or not yet annotated")
                continue
            
            print(f"  ✓ Loaded {len(completed)} annotations")
            print(f"    Annotator: {completed['annotator_id'].iloc[0]}")
            print(f"    Screenshots: {completed['screenshot_id'].nunique()}")
            print(f"    Turns: {len(completed)}")
            
            all_annotations.append(completed)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_annotations:
        print("\n❌ No valid annotations loaded!")
        return None
    
    # Combina
    print(f"\n{'='*70}")
    print("COMBINING ANNOTATIONS")
    print(f"{'='*70}")
    
    merged_df = pd.concat(all_annotations, ignore_index=True)
    
    # Statistiche
    print(f"\nTotal annotations: {len(merged_df)}")
    print(f"Unique screenshots: {merged_df['screenshot_id'].nunique()}")
    print(f"Total turns: {len(merged_df)}")
    
    print(f"\nPer annotator:")
    for annotator in sorted(merged_df['annotator_id'].unique()):
        ann_data = merged_df[merged_df['annotator_id'] == annotator]
        screenshots = ann_data['screenshot_id'].nunique()
        turns = len(ann_data)
        print(f"  {annotator}: {turns} turns, {screenshots} screenshots")
    
    print(f"\nPer subreddit:")
    for sub, count in merged_df['subreddit'].value_counts().items():
        print(f"  {sub}: {count} turns")
    
    # Quality checks
    print(f"\n{'─'*70}")
    print("QUALITY CHECKS")
    print(f"{'─'*70}")
    
    # Check 1: Completezza
    expected_total = 90  # Il tuo totale
    actual = merged_df['screenshot_id'].nunique()
    
    if actual < expected_total:
        missing = expected_total - actual
        print(f"  ⚠️  Missing {missing} screenshots (expected {expected_total}, got {actual})")
    else:
        print(f"  ✓ All {actual} screenshots annotated")
    
    # Check 2: Campi vuoti
    checks = {
        'dialogue_id': merged_df['dialogue_id'].isna().sum(),
        'turn_id': merged_df['turn_id'].isna().sum(),
        'speaker': merged_df['speaker'].isna().sum(),
        'text': merged_df['text'].isna().sum()
    }
    
    has_issues = any(count > 0 for count in checks.values())
    
    if has_issues:
        print(f"  ⚠️  Empty fields found:")
        for field, count in checks.items():
            if count > 0:
                print(f"     - {field}: {count} empty")
    else:
        print(f"  ✓ No empty critical fields")
    
    # Check 3: Speaker values
    valid_speakers = {'User', 'Chatbot'}
    invalid_speakers = merged_df[~merged_df['speaker'].isin(valid_speakers)]
    
    if len(invalid_speakers) > 0:
        print(f"  ⚠️  Invalid speaker values found:")
        for speaker in invalid_speakers['speaker'].unique():
            count = (merged_df['speaker'] == speaker).sum()
            print(f"     - '{speaker}': {count} times")
    else:
        print(f"  ✓ All speakers are valid (User/Chatbot)")
    
    # Salva
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    output_path = Path(output_path)
    
    # CSV
    merged_df.to_csv(output_path, index=False)
    print(f"✓ CSV saved: {output_path}")
    
    # Excel
    excel_path = output_path.with_suffix('.xlsx')
    merged_df.to_excel(excel_path, index=False)
    print(f"✓ Excel saved: {excel_path}")
    
    # Summary stats
    stats_path = output_path.parent / 'merge_summary.txt'
    with open(stats_path, 'w') as f:
        f.write("ANNOTATION MERGE SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total annotations: {len(merged_df)}\n")
        f.write(f"Unique screenshots: {merged_df['screenshot_id'].nunique()}\n")
        f.write(f"Total turns: {len(merged_df)}\n\n")
        
        f.write("Per annotator:\n")
        for annotator in sorted(merged_df['annotator_id'].unique()):
            ann_data = merged_df[merged_df['annotator_id'] == annotator]
            f.write(f"  {annotator}: {len(ann_data)} turns, {ann_data['screenshot_id'].nunique()} screenshots\n")
        
        f.write(f"\nPer subreddit:\n")
        for sub, count in merged_df['subreddit'].value_counts().items():
            f.write(f"  {sub}: {count} turns\n")
    
    print(f"✓ Summary saved: {stats_path}")
    
    print(f"\n{'='*70}")
    print("✅ MERGE COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    
    return merged_df


if __name__ == "__main__":
    merged = merge_annotations_from_folders(
        annotators_dir='./validation_sample/annotators',
        output_path='./validation_sample/gold_standard_annotations.csv'
    )
    
    if merged is not None:
        print("\n🎉 Ready for validation analysis!")
        print("\nOutput files:")
        print("  - gold_standard_annotations.csv")
        print("  - gold_standard_annotations.xlsx")
        print("  - merge_summary.txt")
        
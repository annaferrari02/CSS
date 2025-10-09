import pandas as pd
from pathlib import Path
import shutil

def split_annotation_dataset(
    template_path='./validation_sample/annotation_template.xlsx',
    screenshots_dir='./validation_sample/screenshots',
    output_dir='./validation_sample/annotators',
    n_annotators=3
):
    """
    Divide il dataset di annotazione tra più annotatori
    
    Args:
        template_path: path al template Excel
        screenshots_dir: directory con screenshot
        output_dir: dove creare le cartelle per annotatori
        n_annotators: numero di annotatori
    """
    
    print("="*70)
    print("ANNOTATION DATASET SPLITTER")
    print("="*70)
    
    # Carica template
    print(f"\nLoading template from: {template_path}")
    df = pd.read_excel(template_path)
    total_samples = len(df)
    print(f"Total samples: {total_samples}")
    
    # Dividi equamente
    samples_per_annotator = total_samples // n_annotators
    
    print(f"\nSplit strategy:")
    print(f"  Samples per annotator: ~{samples_per_annotator}")
    
    # Crea directory output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Per ogni annotatore
    for i in range(n_annotators):
        annotator_id = i + 1
        annotator_name = f"annotator_{annotator_id}"
        
        print(f"\n{'─'*70}")
        print(f"Creating package for {annotator_name}")
        
        # Directory annotatore
        ann_dir = output_path / annotator_name
        ann_dir.mkdir(exist_ok=True)
        
        # Sottodirectory screenshots
        ann_screenshots_dir = ann_dir / 'screenshots'
        ann_screenshots_dir.mkdir(exist_ok=True)
        
        # Seleziona screenshot
        start_idx = i * samples_per_annotator
        if i < n_annotators - 1:
            end_idx = start_idx + samples_per_annotator
        else:
            # Ultimo annotatore prende tutto il resto
            end_idx = total_samples
        
        annotator_df = df.iloc[start_idx:end_idx].copy()
        
        # Aggiungi colonna annotator_id pre-compilata
        annotator_df['annotator_id'] = annotator_name
        
        print(f"  Screenshots: {len(annotator_df)} (indices {start_idx}-{end_idx-1})")
        
        # Copia screenshot
        print(f"  Copying screenshots...")
        screenshots_source = Path(screenshots_dir)
        copied = 0
        
        for _, row in annotator_df.iterrows():
            screenshot_path = row['screenshot_path']
            src = Path('./validation_sample') / screenshot_path
            
            if src.exists():
                # Mantieni lo stesso nome
                dst = ann_screenshots_dir / src.name
                shutil.copy2(src, dst)
                copied += 1
                
                # Aggiorna path nel template
                annotator_df.loc[annotator_df['screenshot_id'] == row['screenshot_id'], 'screenshot_path'] = f"screenshots/{src.name}"
        
        print(f"  ✓ Copied {copied} screenshots")
        
        # Salva template Excel
        excel_path = ann_dir / f'{annotator_name}_template.xlsx'
        annotator_df.to_excel(excel_path, index=False, sheet_name='Annotations')
        print(f"  ✓ Template saved: {excel_path.name}")
        
        # Salva anche CSV backup
        csv_path = ann_dir / f'{annotator_name}_template.csv'
        annotator_df.to_csv(csv_path, index=False)
        print(f"  ✓ CSV backup saved: {csv_path.name}")
        
        # Crea istruzioni personalizzate
        instructions = f"""
# ISTRUZIONI PER {annotator_name.upper()}

## Il Tuo Pacchetto

Hai ricevuto **{len(annotator_df)} screenshot** da annotare.

## File nel Pacchetto

1. **{annotator_name}_template.xlsx** ← QUESTO È IL FILE DA COMPILARE
2. **screenshots/** ← cartella con le immagini
3. **INSTRUCTIONS.md** ← questo file

## Come Procedere

### 1. Apri il Template Excel
Apri `{annotator_name}_template.xlsx` con Excel o Google Sheets

### 2. Per Ogni Screenshot

Per ogni riga del file Excel:

a) Apri l'immagine corrispondente dalla cartella `screenshots/`
b) Compila le colonne:
   - **dialogue_id**: assegna ID univoco (es: dial_0001, dial_0002...)
   - **turn_id**: numera i turni dall'alto verso il basso (1, 2, 3...)
   - **speaker**: "User" (destra, chiaro) o "Chatbot" (sinistra, scuro)
   - **text**: trascrivi ESATTAMENTE il testo del bubble
   - **notes**: segnala problemi (low_quality, truncated, etc.)

c) Il campo **annotator_id** è già compilato con "{annotator_name}"

### 3. Linee Guida

**Trascrizione Testo:**
- Copia ESATTAMENTE carattere per carattere
- Mantieni maiuscole, punteggiatura, spaziatura
- Includi emoji: ❤️ 😊 🤔
- Se testo troncato: usa "..."
- Se illeggibile: scrivi in notes "illegible"

**Identificazione Speaker:**
- User: tipicamente DESTRA, sfondo PIÙ CHIARO
- Chatbot: tipicamente SINISTRA, sfondo PIÙ SCURO
- Se incerto: scrivi in notes "ambiguous_speaker"

**Numerazione Turni:**
- Dall'alto verso il basso: 1, 2, 3, 4...
- Bubble consecutive dello STESSO speaker = STESSO turn_id
- Ogni volta che cambia speaker = NUOVO turn_id

**Note Standardizzate:**
- `low_quality` - immagine sfocata/pixelata
- `truncated` - testo tagliato ai bordi
- `overlapping` - bubble sovrapposti
- `ambiguous_speaker` - non chiaro chi parla
- `contains_image` - contiene immagine/gif oltre al testo

### 4. Salvataggio

**IMPORTANTE:**
- Salva il file Excel REGOLARMENTE mentre lavori
- Quando finisci, salva come: `{annotator_name}_COMPLETED.xlsx`
- Invia il file completato via [specificare metodo]

### 5. Tempo Stimato

- Tempo medio: 2-3 minuti per screenshot
- Totale stimato: **{len(annotator_df) * 2.5 / 60:.1f} - {len(annotator_df) * 3.5 / 60:.1f} ore**
- Puoi dividere il lavoro in più sessioni!

### 6. Domande?

Se hai dubbi:
- Controlla ANNOTATION_GUIDELINES.md nella cartella principale
- Contatta: [tua_email@example.com]
- WhatsApp/Telegram: [tuo_contatto]

## Checklist Finale

Prima di inviare, verifica:
- [ ] Tutte le righe sono complete (no celle vuote)
- [ ] dialogue_id è univoco per ogni screenshot
- [ ] turn_id parte da 1 per ogni nuovo dialogo
- [ ] speaker è sempre "User" o "Chatbot" (case-sensitive)
- [ ] text è trascritto esattamente come nell'immagine
- [ ] annotator_id è "{annotator_name}" per tutte le righe
- [ ] File salvato come `{annotator_name}_COMPLETED.xlsx`

## Grazie per il tuo contributo! 🙏

Le tue annotazioni sono fondamentali per validare l'accuratezza 
dell'estrazione automatica dei dialoghi.
"""
        
        instructions_path = ann_dir / 'INSTRUCTIONS.md'
        with open(instructions_path, 'w', encoding='utf-8') as f:
            f.write(instructions)
        print(f"  ✓ Instructions saved: {instructions_path.name}")
    
    # Calcola distribuzione finale
    distribution = []
    for i in range(n_annotators):
        start = i * samples_per_annotator
        end = start + samples_per_annotator if i < n_annotators - 1 else total_samples
        count = end - start
        distribution.append({
            'Annotator': f'Annotator {i+1}',
            'Screenshots': count,
            'Indices': f'{start}-{end-1}',
            'Est. Time': f'{count * 2.5 / 60:.1f}-{count * 3.5 / 60:.1f}h'
        })
    # Crea README generale
    dist_table = "\n".join([
        f"| {d['Annotator']:<12} | {d['Screenshots']:>11} | {d['Indices']:<12} | {d['Est. Time']:<10} |"
        for d in distribution
    ])
    
    main_readme = f"""
# ANNOTATION PROJECT - DISTRIBUTION

## Overview

Dataset di {total_samples} screenshot diviso equamente tra {n_annotators} annotatori.

## Directory Structure
annotators/
├── annotator_1/
│   ├── annotator_1_template.xlsx  ← DA COMPILARE
│   ├── screenshots/                ← {distribution[0]['Screenshots']} immagini
│   └── INSTRUCTIONS.md
├── annotator_2/
│   ├── annotator_2_template.xlsx
│   ├── screenshots/                ← {distribution[1]['Screenshots']} immagini
│   └── INSTRUCTIONS.md
└── annotator_3/
├── annotator_3_template.xlsx
├── screenshots/                ← {distribution[2]['Screenshots']} immagini
└── INSTRUCTIONS.md

## Distribution Summary

| Annotator    | Screenshots | Indices      | Est. Time  |
|--------------|-------------|--------------|------------|
{dist_table}

**Total: {total_samples} screenshots**

## Instructions for Project Manager

### 1. Distribution
Send each annotator their respective folder:
- `annotator_1/` → Annotator 1
- `annotator_2/` → Annotator 2
- `annotator_3/` → Annotator 3

### 2. Remind Annotators
- Complete annotations independently
- Follow INSTRUCTIONS.md in their folder
- Save as `annotatorX_COMPLETED.xlsx` when done

### 3. Collection
When done, collect:
- `annotator_1_COMPLETED.xlsx`
- `annotator_2_COMPLETED.xlsx`
- `annotator_3_COMPLETED.xlsx`

### 4. Next Steps
After collection:
1. Combine all annotations
2. Run validation analysis
3. Calculate accuracy metrics

## Contact

Project Lead: [your_email@example.com]
Questions: [contact_method]
"""
    
    readme_path = output_path / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(main_readme)
    
    print(f"\n{'='*70}")
    print("✅ SPLIT COMPLETED!")
    print(f"{'='*70}")
    print(f"\nDistribution:")
    for d in distribution:
        print(f"  {d['Annotator']}: {d['Screenshots']} screenshots ({d['Indices']}) - {d['Est. Time']}")
    
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"\nNext steps:")
    print(f"  1. Review annotators/ folder")
    print(f"  2. Send each folder to respective annotator")
    print(f"  3. Collect completed files")
    print(f"\n📧 Distribution:")
    for i in range(n_annotators):
        print(f"  - annotator_{i+1}/ → Send to Annotator {i+1}")


if __name__ == "__main__":
    split_annotation_dataset(
        template_path='./validation_sample/annotation_template.xlsx',
        screenshots_dir='./validation_sample/screenshots',
        output_dir='./validation_sample/annotators',
        n_annotators=3
    )
    

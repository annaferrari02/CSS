
# ISTRUZIONI PER ANNOTATOR_3

## Il Tuo Pacchetto

Hai ricevuto **30 screenshot** da annotare.

## File nel Pacchetto

1. **annotator_3_template.xlsx** ← QUESTO È IL FILE DA COMPILARE
2. **screenshots/** ← cartella con le immagini
3. **INSTRUCTIONS.md** ← questo file

## Come Procedere

### 1. Apri il Template Excel
Apri `annotator_3_template.xlsx` con Excel o Google Sheets

### 2. Per Ogni Screenshot

Per ogni riga del file Excel:

a) Apri l'immagine corrispondente dalla cartella `screenshots/`
b) Compila le colonne:
   - **dialogue_id**: assegna ID univoco (es: dial_0001, dial_0002...)
   - **turn_id**: numera i turni dall'alto verso il basso (1, 2, 3...)
   - **speaker**: "User" (destra, chiaro) o "Chatbot" (sinistra, scuro)
   - **text**: trascrivi ESATTAMENTE il testo del bubble
   - **notes**: segnala problemi (low_quality, truncated, etc.)

c) Il campo **annotator_id** è già compilato con "annotator_3"

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
- Quando finisci, salva come: `annotator_3_COMPLETED.xlsx`
- Invia il file completato via [specificare metodo]

### 5. Tempo Stimato

- Tempo medio: 2-3 minuti per screenshot
- Totale stimato: **1.2 - 1.8 ore**
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
- [ ] annotator_id è "annotator_3" per tutte le righe
- [ ] File salvato come `annotator_3_COMPLETED.xlsx`

## Grazie per il tuo contributo! 🙏

Le tue annotazioni sono fondamentali per validare l'accuratezza 
dell'estrazione automatica dei dialoghi.

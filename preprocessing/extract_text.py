import cv2
import numpy as np
import pytesseract
import re
from pathlib import Path
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from typing import List, Dict, Tuple, Callable
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurazione Tesseract (modifica il path se necessario)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==================== PARAMETRI GLOBALI ====================
LINE_GROUPING_THRESHOLD = 15  # Pixel tolerance per raggruppare parole sulla stessa riga
SINGLE_SPEAKER_THRESHOLD = 0.15  # Threshold per rilevare conversazioni single-speaker
CONFIDENCE_THRESHOLD = 35  # Soglia minima di confidenza OCR (più basso = più permissivo)
MIN_BUBBLE_AREA = 1000  # Area minima per rilevare una bubble
POSITION_THRESHOLD = 0.55  # Threshold per determinare speaker da posizione (fallback)

# Paths di default
BASE_DIR = Path(__file__).parent.parent
SCREENSHOTS_DIR = BASE_DIR / 'data' / 'screenshots'
OUTPUT_FILE = BASE_DIR / 'data' / 'extracted_chat.txt'


# ==================== PRE-PROCESSING AVANZATO ====================

def preprocess_image_advanced(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-processing multi-stage per massimizzare accuratezza OCR
    Restituisce sia l'immagine processata che una versione per bubble detection
    """
    # Converti in scala di grigi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. RIDUZIONE RUMORE con filtro bilaterale (mantiene i bordi)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. CLAHE per migliorare contrasto locale
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 3. SHARPENING per testo più nitido
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
    
    # 4. BINARIZZAZIONE ADATTIVA (migliore per sfondi variabili)
    binary = cv2.adaptiveThreshold(
        sharpened, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=11, 
        C=2
    )
    
    # 5. MORPHOLOGICAL OPERATIONS per pulire artefatti
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Versione per bubble detection (usa immagine originale con blur)
    bubble_img = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return cleaned, bubble_img


# ==================== BUBBLE DETECTION ====================

def detect_chat_bubbles(img: np.ndarray, bubble_img: np.ndarray) -> List[Dict]:
    """
    Rileva le chat bubbles usando analisi dei contorni e colore
    Versione migliorata con doppia strategia
    """
    height, width = img.shape[:2]
    
    # Strategia 1: Threshold adattivo
    thresh = cv2.adaptiveThreshold(
        bubble_img, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=15, 
        C=10
    )
    
    # Morfologia per unire regioni frammentate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel, iterations=2)
    
    # Trova contorni
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubbles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filtri per identificare bubble valide
        is_valid_size = (area > MIN_BUBBLE_AREA and 
                        w > 50 and h > 20 and 
                        w < width * 0.85 and h < height * 0.6)
        
        is_in_valid_region = (y > height * 0.02 and y < height * 0.98)
        
        # Aspect ratio ragionevole (non troppo stretto/largo)
        aspect_ratio = w / h if h > 0 else 0
        is_reasonable_shape = 0.5 < aspect_ratio < 8
        
        if is_valid_size and is_in_valid_region and is_reasonable_shape:
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Determina lato (con margine più ampio per gestire layouts diversi)
            side = 'right' if center_x > width * 0.45 else 'left'
            
            bubbles.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'center_x': center_x,
                'center_y': center_y,
                'side': side,
                'area': area
            })
    
    logger.debug(f"Detected {len(bubbles)} chat bubbles")
    return bubbles


# ==================== OCR E ESTRAZIONE TESTO ====================

def extract_text_with_ocr(img: np.ndarray, processed_img: np.ndarray) -> List[Dict]:
    """
    Estrae testo con OCR ottimizzato usando configurazioni multiple
    """
    # Configurazioni OCR da provare (in ordine di priorità)
    configs = [
        '--oem 3 --psm 6',  # PSM 6 = blocco uniforme di testo
        '--oem 3 --psm 4',  # PSM 4 = singola colonna di testo
        '--oem 3 --psm 3',  # PSM 3 = automatic page segmentation
    ]
    
    best_words = []
    max_confidence = 0
    
    for config in configs:
        try:
            data = pytesseract.image_to_data(
                processed_img, 
                output_type=pytesseract.Output.DICT,
                config=config
            )
            
            # Estrai parole con confidenza alta
            words = [
                {
                    'text': data['text'][i].strip(),
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i],
                    'conf': int(data['conf'][i])
                }
                for i in range(len(data['text']))
                if int(data['conf'][i]) > CONFIDENCE_THRESHOLD 
                and len(data['text'][i].strip()) > 0
            ]
            
            # Calcola confidenza media
            if words:
                avg_conf = np.mean([w['conf'] for w in words])
                if avg_conf > max_confidence:
                    max_confidence = avg_conf
                    best_words = words
                    
        except Exception as e:
            logger.warning(f"OCR config failed: {config} - {e}")
            continue
    
    logger.debug(f"Extracted {len(best_words)} words with avg confidence {max_confidence:.1f}")
    return best_words


# ==================== RAGGRUPPAMENTO PAROLE IN RIGHE ====================

def group_words_into_lines(words_data: List[Dict], threshold: int = LINE_GROUPING_THRESHOLD) -> List[Dict]:
    """
    Raggruppa parole in righe basandosi sulla posizione Y
    Versione ottimizzata con gestione di casi edge
    """
    if not words_data:
        return []
    
    # Ordina per Y prima, poi per X
    words_data.sort(key=lambda x: (x['y'], x['x']))
    
    lines = []
    current_line = {
        'words': [words_data[0]], 
        'y': words_data[0]['y'],
        'y_min': words_data[0]['y'],
        'y_max': words_data[0]['y'] + words_data[0]['h']
    }
    
    for word in words_data[1:]:
        word_y_center = word['y'] + word['h'] / 2
        line_y_center = (current_line['y_min'] + current_line['y_max']) / 2
        
        # Controllo sovrapposizione verticale
        overlaps = (word['y'] <= current_line['y_max'] + threshold and 
                   word['y'] + word['h'] >= current_line['y_min'] - threshold)
        
        same_line = abs(word_y_center - line_y_center) <= threshold or overlaps
        
        if same_line:
            current_line['words'].append(word)
            current_line['y_min'] = min(current_line['y_min'], word['y'])
            current_line['y_max'] = max(current_line['y_max'], word['y'] + word['h'])
        else:
            # Salva riga corrente e inizia nuova
            if current_line['words']:
                current_line['words'].sort(key=lambda x: x['x'])
                lines.append(current_line)
            
            current_line = {
                'words': [word], 
                'y': word['y'],
                'y_min': word['y'],
                'y_max': word['y'] + word['h']
            }
    
    # Aggiungi ultima riga
    if current_line['words']:
        current_line['words'].sort(key=lambda x: x['x'])
        lines.append(current_line)
    
    return lines


# ==================== DETERMINAZIONE SPEAKER ====================

def create_speaker_classifier(lines: List[Dict], width: int, bubbles: List[Dict]) -> Callable:
    """
    Crea un classificatore per determinare lo speaker usando strategia ibrida:
    1. Clustering delle posizioni X
    2. Bubble detection
    3. Fallback basato su posizione semplice
    """
    if not lines:
        return lambda x, y: "Chatbot:"
    
    # Estrai posizioni di inizio riga
    start_positions = [line['words'][0]['x'] for line in lines if line['words']]
    
    if len(start_positions) < 2:
        return lambda x, y: "Chatbot:"
    
    # STRATEGIA 1: Clustering K-means
    X = np.array(start_positions).reshape(-1, 1)
    
    try:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
        centers = sorted(kmeans.cluster_centers_.flatten())
        left_center, right_center = centers[0], centers[1]
        
        # Se i cluster sono troppo vicini → single speaker
        if (right_center - left_center) < (width * SINGLE_SPEAKER_THRESHOLD):
            logger.info("Single speaker conversation detected")
            return lambda x, y: "Chatbot:"
        
        threshold_cluster = left_center + (right_center - left_center) / 2
        
    except Exception as e:
        logger.warning(f"Clustering failed: {e}")
        threshold_cluster = width * POSITION_THRESHOLD
    
    # STRATEGIA 2: Bubble-based classification
    def classify_with_bubbles(x: float, y: float) -> str:
        # Prova prima con bubble detection
        for bubble in bubbles:
            if (bubble['x'] - 30 <= x <= bubble['x'] + bubble['w'] + 30 and
                bubble['y'] - 30 <= y <= bubble['y'] + bubble['h'] + 30):
                return "USER:" if bubble['side'] == 'right' else "Chatbot:"
        
        # Fallback: usa clustering
        return "USER:" if x > threshold_cluster else "Chatbot:"
    
    return classify_with_bubbles


# ==================== PULIZIA TESTO ====================

# Regex pre-compilate per performance
TIMESTAMP_PATTERN = re.compile(r'\b\d{1,2}:\d{2}(?:\s?[AP]M)?\b')
ARTIFACTS_PATTERN = re.compile(r'[|\\/_#$%^&*+=<>{}[\]]+')
MENTION_PATTERN = re.compile(r'^@\w+\s*')
MULTI_SPACE_PATTERN = re.compile(r'\s+')
PUNCT_SPACE_PATTERN = re.compile(r'\s+([.,!?;:])')
EMOJI_ARTIFACTS = re.compile(r'[\x00-\x1f\x7f-\x9f]')

def clean_extracted_text(text: str) -> str:
    """
    Pulizia avanzata del testo estratto con focus su preservare parole valide
    """
    if not text:
        return ""
    
    # Normalizza spazi
    text = MULTI_SPACE_PATTERN.sub(' ', text)
    
    # Rimuovi timestamps
    text = TIMESTAMP_PATTERN.sub('', text)
    
    # Rimuovi artefatti comuni OCR (ma preserva punteggiatura normale)
    text = ARTIFACTS_PATTERN.sub('', text)
    
    # Rimuovi menzioni
    text = MENTION_PATTERN.sub('', text)
    
    # Rimuovi caratteri di controllo
    text = EMOJI_ARTIFACTS.sub('', text)
    
    # Sistema spazi intorno a punteggiatura
    text = PUNCT_SPACE_PATTERN.sub(r'\1', text)
    
    # Rimuovi spazi multipli
    text = MULTI_SPACE_PATTERN.sub(' ', text)
    
    # Rimuovi spazi iniziali/finali
    text = text.strip()
    
    # Post-processing: rimuovi righe con solo numeri/simboli
    if text and len(text) > 1:
        # Conta lettere reali
        letter_count = sum(c.isalpha() for c in text)
        if letter_count < len(text) * 0.3 and len(text) > 3:
            # Troppi simboli, probabilmente artefatto
            logger.debug(f"Filtered low-quality text: '{text}'")
            return ""
    
    return text


# ==================== ESTRAZIONE PRINCIPALE ====================

def extract_chat_from_image(image_path: Path) -> str:
    """
    Pipeline completa di estrazione chat da immagine
    Combina pre-processing, bubble detection, OCR e classificazione speaker
    """
    try:
        # Carica immagine
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        height, width = img.shape[:2]
        logger.info(f"Processing {image_path.name} ({width}x{height})")
        
        # 1. PRE-PROCESSING
        processed_img, bubble_img = preprocess_image_advanced(img)
        
        # 2. BUBBLE DETECTION
        bubbles = detect_chat_bubbles(img, bubble_img)
        
        # 3. OCR
        words_data = extract_text_with_ocr(img, processed_img)
        
        if not words_data:
            logger.warning(f"No text detected in {image_path.name}")
            return "No text detected in image"
        
        # 4. RAGGRUPPA IN RIGHE
        lines = group_words_into_lines(words_data)
        
        if not lines:
            logger.warning(f"No text lines formed in {image_path.name}")
            return "No text lines found"
        
        # 5. CREA CLASSIFICATORE SPEAKER
        speaker_classifier = create_speaker_classifier(lines, width, bubbles)
        
        # 6. COSTRUISCI MESSAGGI
        messages = []
        for line in lines:
            if not line['words']:
                continue
            
            # Posizione rappresentativa della riga
            line_x = line['words'][0]['x']
            line_y = line['y']
            
            # Determina speaker
            speaker = speaker_classifier(line_x, line_y)
            
            # Combina testo
            text = ' '.join(word['text'] for word in line['words'])
            text = clean_extracted_text(text)
            
            if text:
                messages.append({
                    'speaker': speaker,
                    'text': text,
                    'y': line_y,
                    'confidence': np.mean([w['conf'] for w in line['words']])
                })
        
        # Ordina per posizione verticale
        messages.sort(key=lambda x: x['y'])
        
        # Filtra messaggi duplicati consecutivi
        filtered_messages = []
        prev_text = None
        for msg in messages:
            if msg['text'] != prev_text:
                filtered_messages.append(f"{msg['speaker']} {msg['text']}")
                prev_text = msg['text']
        
        result = '\n\n'.join(filtered_messages)
        logger.info(f"Extracted {len(filtered_messages)} messages from {image_path.name}")
        
        return result
        
    except Exception as e:
        logger.error(f"Extraction error for {image_path}: {str(e)}")
        return f"Extraction error: {str(e)}"


# ==================== PROCESSING PARALLELO ====================

def process_single_image(image_path: Path) -> Dict:
    """Wrapper per processing singola immagine (per parallelizzazione)"""
    extracted_text = extract_chat_from_image(image_path)
    
    return {
        'image_path': str(image_path),
        'filename': image_path.name,
        'relative_path': str(image_path.relative_to(SCREENSHOTS_DIR)) if image_path.is_relative_to(SCREENSHOTS_DIR) else image_path.name,
        'extracted_text': extracted_text
    }


def process_all_screenshots(
    input_dir: Path = SCREENSHOTS_DIR,
    use_parallel: bool = True, 
    max_workers: int = None
) -> List[Dict]:
    """
    Processa tutte le immagini in una directory
    
    Args:
        input_dir: Directory contenente gli screenshot
        use_parallel: Se True, usa multiprocessing
        max_workers: Numero di worker paralleli (default: CPU count - 1)
    
    Returns:
        Lista di dizionari con risultati estrazione
    """
    if not input_dir.exists():
        logger.error(f"Folder '{input_dir}' not found")
        return []
    
    # Trova tutte le immagini (ricorsivo)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(input_dir.glob(f'**/{ext}'))
    
    if not image_files:
        logger.warning(f"No image files found in '{input_dir}'")
        return []
    
    logger.info(f"Found {len(image_files)} images in '{input_dir}'")
    print("=" * 80)
    
    results = []
    
    # Parallelo per > 3 immagini
    if use_parallel and len(image_files) > 3:
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        
        logger.info(f"Using {max_workers} parallel workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_img = {
                executor.submit(process_single_image, img_path): img_path 
                for img_path in image_files
            }
            
            for i, future in enumerate(as_completed(future_to_img), 1):
                img_path = future_to_img[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✓ [{i}/{len(image_files)}] {img_path.name}")
                except Exception as e:
                    logger.error(f"Error processing {img_path.name}: {e}")
                    print(f"✗ [{i}/{len(image_files)}] {img_path.name} - ERROR")
    
    else:
        # Sequenziale
        logger.info("Using sequential processing")
        for i, image_path in enumerate(image_files, 1):
            print(f"Processing [{i}/{len(image_files)}]: {image_path.name}")
            try:
                result = process_single_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error: {e}")
    
    # Mostra esempi
    print("\n" + "=" * 80)
    print("EXTRACTION EXAMPLES")
    print("=" * 80)
    
    for i, result in enumerate(results[:2], 1):
        print(f"\n--- Example {i}: {result['filename']} ---")
        preview = result['extracted_text'][:400]
        print(preview)
        if len(result['extracted_text']) > 400:
            print("... (truncated)")
    
    print("=" * 80)
    
    return results


# ==================== SALVATAGGIO RISULTATI ====================

def save_results(results: List[Dict], output_file: Path = OUTPUT_FILE):
    """Salva risultati in file di testo formattato"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EXTRACTED CHAT CONVERSATIONS FROM SCREENSHOTS\n")
        f.write(f"Total images processed: {len(results)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"IMAGE {i}/{len(results)}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Filename: {result['filename']}\n")
            f.write(f"Location: {result['relative_path']}\n")
            f.write(f"{'-' * 80}\n\n")
            f.write(result['extracted_text'])
            f.write(f"\n\n{'=' * 80}\n")
    
    logger.info(f"Results saved to: {output_file}")


# ==================== MAIN ====================

def main():
    """Entry point principale"""
    print("=" * 80)
    print("UNIFIED CHAT TEXT EXTRACTOR (OPTIMIZED)")
    print("=" * 80)
    print(f"Input folder: {SCREENSHOTS_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 80)
    
    # Process con parallelizzazione automatica
    results = process_all_screenshots(
        input_dir=SCREENSHOTS_DIR,
        use_parallel=True
    )
    
    if results:
        print(f"\n✅ Successfully processed {len(results)} images")
        save_results(results, OUTPUT_FILE)
        print("\n✅ Extraction completed!")
        print(f"📄 Results available at: {OUTPUT_FILE}")
    else:
        print("\n⚠️ No images were processed")


if __name__ == "__main__":
    main()
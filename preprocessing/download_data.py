import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import json
import requests
import os
from urllib.parse import urlparse
import time
import cv2
import numpy as np
from PIL import Image
import re
from io import BytesIO
from pathlib import Path

print("Current directory:", os.getcwd())

# Percorsi relativi alla nuova struttura
BASE_DIR = Path(__file__).parent.parent  # CSS/
DATA_DIR = BASE_DIR / 'data'
SCREENSHOTS_DIR = DATA_DIR / 'screenshots'

# Crea le directory se non esistono
DATA_DIR.mkdir(exist_ok=True)
SCREENSHOTS_DIR.mkdir(exist_ok=True)


def process_all_json_files():
    """
    Processa tutti i file JSON trovati nella cartella data/ e sottocartelle
    """
    # Trova tutti i file .json e .jsonl in data/
    json_files = list(DATA_DIR.glob('**/*.json')) + list(DATA_DIR.glob('**/*.jsonl'))
    
    if not json_files:
        print(f"Nessun file JSON trovato in {DATA_DIR}")
        return
    
    print(f"Trovati {len(json_files)} file JSON da processare")
    print("="*60)
    
    total_processed = 0
    total_downloaded = 0
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        print("-"*60)
        
        # Determina il nome della sottocartella per le immagini
        # Es: data/AIRelationship/file.json -> screenshots/AIRelationship/
        relative_path = json_file.parent.relative_to(DATA_DIR)
        output_subdir = SCREENSHOTS_DIR / relative_path / json_file.stem
        
        try:
            processed, downloaded = process_reddit_posts_with_images(
                json_file, 
                output_subdir
            )
            total_processed += processed
            total_downloaded += downloaded
        except Exception as e:
            print(f"Errore processando {json_file.name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("RIEPILOGO TOTALE")
    print("="*60)
    print(f"Post processati: {total_processed}")
    print(f"Immagini scaricate: {total_downloaded}")
    print(f"Immagini salvate in: {SCREENSHOTS_DIR}")
    print("="*60)


def process_reddit_posts_with_images(json_file_path, output_dir):
    """
    Process Reddit posts from a JSON/JSONL file
    
    Args:
        json_file_path: Path object al file JSON/JSONL
        output_dir: Path object alla directory di output
    
    Returns:
        tuple: (processed_count, downloaded_count)
    """
    # Crea directory output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input: {json_file_path}")
    print(f"Output: {output_dir}")
    
    processed_count = 0
    downloaded_count = 0
    
    # Leggi il file (supporta sia JSON che JSONL)
    with open(json_file_path, 'r', encoding='utf-8') as f:
        if json_file_path.suffix == '.jsonl':
            # JSONL: una riga = un oggetto JSON
            for line_num, line in enumerate(f, start=1):
                try:
                    post = json.loads(line.strip())
                    if has_images(post):
                        processed_count += 1
                        num_images = download_post_images(post, output_dir)
                        downloaded_count += num_images
                except json.JSONDecodeError as e:
                    print(f"  Errore parsing linea {line_num}: {e}")
                    continue
        else:
            # JSON: singolo oggetto o array
            try:
                data = json.load(f)
                # Se è un array, processa ogni elemento
                posts = data if isinstance(data, list) else [data]
                for post in posts:
                    if has_images(post):
                        processed_count += 1
                        num_images = download_post_images(post, output_dir)
                        downloaded_count += num_images
            except json.JSONDecodeError as e:
                print(f"  Errore parsing JSON: {e}")
    
    print(f"  Post processati: {processed_count}")
    print(f"  Immagini scaricate: {downloaded_count}")
    
    return processed_count, downloaded_count


def download_post_images(post, output_dir):
    """
    Scarica tutte le immagini con testo da un singolo post
    
    Returns:
        int: numero di immagini scaricate
    """
    media_metadata = post.get('media_metadata', {})
    post_id = post.get('id', 'unknown')
    downloaded = 0
    
    for idx, (media_id, metadata) in enumerate(media_metadata.items(), start=1):
        if metadata.get('status') == 'valid' and metadata.get('e') == 'Image':
            # Get preview URL
            preview_url = get_preview_url(metadata)
            
            if preview_url and prescreen_image_for_text(preview_url, media_id):
                # Download full resolution
                full_url = metadata['s']['u']
                filename = f"{post_id}_{idx}_{media_id}.jpg"
                
                if download_image(full_url, filename, output_dir):
                    downloaded += 1
                
                time.sleep(0.5)  # Rate limiting
    
    return downloaded


def get_preview_url(metadata):
    """Get medium-resolution preview URL"""
    previews = metadata.get('p', [])
    for preview in previews:
        if 320 <= preview.get('x', 0) <= 640:
            return preview.get('u')
    return previews[0].get('u') if previews else None


def prescreen_image_for_text(preview_url, media_id):
    """Check if image contains text before downloading"""
    try:
        if '&amp;' in preview_url:
            preview_url = preview_url.replace('&amp;', '&')
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(preview_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        text_data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
        
        valid_words = 0
        for i in range(len(text_data['text'])):
            if int(text_data['conf'][i]) > 30 and len(text_data['text'][i].strip()) > 1:
                valid_words += 1
        
        return valid_words >= 3
        
    except Exception as e:
        print(f"    Pre-screen failed for {media_id}: {e}")
        return False


def has_images(post):
    """Check if post contains images"""
    return (post.get('is_gallery', False) and 
            'media_metadata' in post and 
            post.get('media_metadata'))


def download_image(url, filename, output_dir):
    """Download image to file"""
    try:
        if '&amp;' in url:
            url = url.replace('&amp;', '&')
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        image_path = Path(output_dir) / filename
        
        with open(image_path, 'wb') as f:
            f.write(response.content)
        
        print(f"    Downloaded: {filename}")
        return True
        
    except Exception as e:
        print(f"    Error downloading {filename}: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("REDDIT IMAGE DOWNLOADER")
    print("="*60)
    print(f"Searching for JSON files in: {DATA_DIR}")
    print(f"Images will be saved to: {SCREENSHOTS_DIR}")
    print("="*60)
    
    process_all_json_files()
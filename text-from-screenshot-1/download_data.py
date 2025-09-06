import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

import json
import requests
import os
from urllib.parse import urlparse
import time
import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
from io import BytesIO

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_reddit_posts_with_images(jsonl_file_path, output_dir="downloaded_images"):
    """
    Process Reddit posts, pre-screen images for text content, and only download/process those with chat messages
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    processed_count = 0
    downloaded_count = 0
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                post = json.loads(line.strip())
                
                if has_images(post):
                    print(f"Processing post {line_num + 1}: {post.get('id', 'unknown')}")
                    processed_count += 1
                    
                    # Extract post metadata
                    post_metadata = extract_post_metadata(post)
                    
                    # Pre-screen and process images
                    image_results = prescreen_and_process_images(post, output_dir)
                    
                    if image_results:
                        downloaded_count += len(image_results)
                        results.append({
                            'post_metadata': post_metadata,
                            'image_analysis': image_results
                        })
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num + 1}: {e}")
                continue
    
    print(f"\nSummary: Processed {processed_count} posts with images, downloaded {downloaded_count} images with text")
    return results

def prescreen_and_process_images(post, output_dir):
    """
    Pre-screen images for text content before downloading
    """
    results = []
    media_metadata = post.get('media_metadata', {})
    
    for media_id, metadata in media_metadata.items():
        if metadata.get('status') == 'valid' and metadata.get('e') == 'Image':
            # Get medium resolution URL for pre-screening
            preview_url = get_preview_url(metadata)
            
            if preview_url:
                # Pre-screen for text content
                has_text = prescreen_image_for_text(preview_url, media_id)
                
                if has_text:
                    print(f"  ✓ Image {media_id} contains text - downloading full resolution...")
                    
                    # Download full resolution image
                    full_url = metadata['s']['u']
                    image_path = download_image(full_url, media_id, output_dir)
                    
                    if image_path:
                        # Extract chat text from full resolution image
                        extracted_text = extract_chat_from_image(image_path)
                        
                        results.append({
                            'media_id': media_id,
                            'image_url': full_url,
                            'local_path': image_path,
                            'extracted_chat': extracted_text
                        })
                else:
                    print(f"  ✗ Image {media_id} has no significant text - skipping download")
                
                # Pause between requests
                time.sleep(0.5)
    
    return results

def get_preview_url(metadata):
    """
    Get a medium-resolution preview URL for pre-screening
    """
    previews = metadata.get('p', [])
    
    # Look for medium resolution (around 320-640px width)
    for preview in previews:
        if 320 <= preview.get('x', 0) <= 640:
            return preview.get('u')
    
    # Fallback to smallest available preview
    if previews:
        return previews[0].get('u')
    
    return None

def prescreen_image_for_text(preview_url, media_id):
    """
    Pre-screen image to determine if it contains significant text content
    """
    try:
        # Clean URL
        if '&amp;' in preview_url:
            preview_url = preview_url.replace('&amp;', '&')
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Download preview image to memory
        response = requests.get(preview_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Load image from memory
        image = Image.open(BytesIO(response.content))
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Quick text detection using OCR
        text_data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
        
        # Count words with reasonable confidence
        valid_words = 0
        total_text_length = 0
        
        for i in range(len(text_data['text'])):
            confidence = int(text_data['conf'][i])
            text = text_data['text'][i].strip()
            
            if confidence > 30 and len(text) > 1:
                valid_words += 1
                total_text_length += len(text)
        
        # Determine if image likely contains chat messages
        # Criteria: at least 3 words with decent confidence and reasonable total length
        has_sufficient_text = (valid_words >= 3 and total_text_length >= 10)
        
        # Additional check: look for chat-like patterns
        all_text = ' '.join([text_data['text'][i] for i in range(len(text_data['text'])) 
                           if int(text_data['conf'][i]) > 30])
        
        # Look for chat indicators (messages, conversation patterns, etc.)
        chat_indicators = ['message', 'chat', 'conversation', 'reply', 'says', 'said', 
                          ':', 'hi', 'hello', 'how are', 'what', 'why', 'when', 'where']
        
        has_chat_patterns = any(indicator.lower() in all_text.lower() 
                               for indicator in chat_indicators)
        
        return has_sufficient_text and (has_chat_patterns or valid_words >= 5)
        
    except Exception as e:
        print(f"    Warning: Pre-screen failed for {media_id}: {e}")
        # If pre-screening fails, err on the side of including the image
        return True

def has_images(post):
    """
    Check if a Reddit post contains images
    """
    return (post.get('is_gallery', False) and 
            'media_metadata' in post and 
            post.get('media_metadata'))

def extract_post_metadata(post):
    """
    Extract relevant metadata from Reddit post
    """
    return {
        'id': post.get('id'),
        'author': post.get('author'),
        'subreddit': post.get('subreddit'),
        'title': post.get('title', ''),
        'selftext': post.get('selftext', ''),
        'created_utc': post.get('created_utc'),
        'score': post.get('score', 0),
        'num_comments': post.get('num_comments', 0),
        'permalink': post.get('permalink', '')
    }

def download_image(url, media_id, output_dir):
    """
    Download an image given its URL
    """
    try:
        if '&amp;' in url:
            url = url.replace('&amp;', '&')
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        image_path = os.path.join(output_dir, f"{media_id}.jpg")
        with open(image_path, 'wb') as f:
            f.write(response.content)
        
        return image_path
        
    except Exception as e:
        print(f"    Error downloading image {media_id}: {e}")
        return None

def extract_chat_from_image(image_path):
    """
    Extract chat conversation text from image
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return f"Error: cannot load image {image_path}"
        
        height, width = img.shape[:2]
        
        # Apply OCR
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        # Extract words with positions
        words_data = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30 and data['text'][i].strip():
                words_data.append({
                    'text': data['text'][i],
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i]
                })
        
        if not words_data:
            return "No text detected"
        
        # Group into lines
        lines = group_words_into_lines(words_data)
        
        # Determine speaker for each line
        messages = []
        for line in lines:
            center_x = np.mean([word['x'] + word['w']/2 for word in line['words']])
            is_right = center_x > width * 0.5
            
            text = ' '.join([word['text'] for word in line['words']])
            text = clean_extracted_text(text)
            
            if text.strip():
                speaker = "USER:" if is_right else "Chatbot:"
                messages.append({
                    'speaker': speaker,
                    'text': text,
                    'y': line['y']
                })
        
        # Sort top to bottom
        messages.sort(key=lambda x: x['y'])
        
        # Format output
        result = []
        for msg in messages:
            result.append(f"{msg['speaker']} {msg['text']}")
        
        return '\n\n'.join(result)
        
    except Exception as e:
        return f"Extraction error: {str(e)}"

def group_words_into_lines(words_data, line_threshold=10):
    """
    Group words into lines based on Y position
    """
    if not words_data:
        return []
    
    words_data.sort(key=lambda x: x['y'])
    
    lines = []
    current_line = {'words': [words_data[0]], 'y': words_data[0]['y']}
    
    for word in words_data[1:]:
        if abs(word['y'] - current_line['y']) <= line_threshold:
            current_line['words'].append(word)
        else:
            if current_line['words']:
                lines.append(current_line)
            current_line = {'words': [word], 'y': word['y']}
    
    if current_line['words']:
        lines.append(current_line)
    
    # Sort words by X in each line
    for line in lines:
        line['words'].sort(key=lambda x: x['x'])
    
    return lines

def clean_extracted_text(text):
    """
    Clean extracted text
    """
    text = re.sub(r'[^\w\s.,!?\'"()-]', '', text)
    text = re.sub(r'\b\d{1,2}:\d{2}(?:\s?[AP]M)?\b', '', text)
    text = re.sub(r'^@\w+\s*', '', text)
    text = ' '.join(text.split())
    return text.strip()

def save_results(results, output_file="chat_extraction_results.json"):
    """
    Save results to JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")

# Main usage
if __name__ == "__main__":
    jsonl_file = "C:/Users/anna2/OneDrive/Desktop/DS1/Second semester/CSS/project/r_Replika_posts.jsonl"  # Your Reddit file
    
    print("Starting processing of Reddit posts with text pre-screening...")
    results = process_reddit_posts_with_images(jsonl_file)
    
    print(f"\nProcessed {len(results)} posts with text-containing images")
    
    # Save results
    save_results(results)
    
    # Print example
    if results:
        print("\nExample extracted conversation:")
        print("="*50)
        first_result = results[0]
        print(f"Post: {first_result['post_metadata']['title']}")
        print(f"Author: {first_result['post_metadata']['author']}")
        print("Extracted conversation:")
        if first_result['image_analysis']:
            print(first_result['image_analysis'][0]['extracted_chat'])


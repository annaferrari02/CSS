import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import os
import glob

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def process_downloaded_images(images_folder="downloaded_images"):
    """
    Process all images in the downloaded_images folder
    """
    if not os.path.exists(images_folder):
        print(f"Error: Folder '{images_folder}' not found")
        return []
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(images_folder, ext)))
    
    if not image_files:
        print(f"No image files found in '{images_folder}'")
        return []
    
    print(f"Found {len(image_files)} images in '{images_folder}'")
    
    results = []
    
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        extracted_text = extract_chat_with_bubble_detection(image_path)
        
        results.append({
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'extracted_text': extracted_text
        })
        
        # Show first example
        if i == 0:
            print("\n" + "="*60)
            print("EXAMPLE - FIRST IMAGE EXTRACTION:")
            print("="*60)
            print(f"Image: {os.path.basename(image_path)}")
            print("\nExtracted conversation:")
            print("-" * 40)
            print(extracted_text)
            print("="*60)
    
    return results

def extract_chat_with_bubble_detection(image_path):
    """
    Extract chat using simple OCR approach but with bubble detection for speaker identification
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return f"Error: cannot load image {image_path}"
        
        height, width = img.shape[:2]
        
        # Apply OCR to get all text with positions
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        # Extract words with their positions and confidence
        words_data = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30 and data['text'][i].strip():
                words_data.append({
                    'text': data['text'][i],
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i],
                    'conf': data['conf'][i]
                })
        
        if not words_data:
            return "No text detected in image"
        
        # Detect bubble regions using color/contour analysis
        bubble_regions = detect_bubble_regions(img)
        
        # Group words into lines
        lines = group_words_into_lines(words_data)
        
        # Assign each line to a speaker based on bubble detection
        messages = []
        for line in lines:
            # Get the center position of the line
            line_center_x = np.mean([word['x'] + word['w']/2 for word in line['words']])
            line_center_y = np.mean([word['y'] + word['h']/2 for word in line['words']])
            
            # Determine speaker based on bubble position or fallback to simple position
            speaker = determine_speaker(line_center_x, line_center_y, width, bubble_regions)
            
            # Combine words into text
            text = ' '.join([word['text'] for word in line['words']])
            text = clean_extracted_text(text)
            
            if text.strip():
                messages.append({
                    'speaker': speaker,
                    'text': text,
                    'y': line['y']
                })
        
        # Sort messages by vertical position (top to bottom)
        messages.sort(key=lambda x: x['y'])
        
        # Format output
        result = []
        for msg in messages:
            result.append(f"{msg['speaker']} {msg['text']}")
        
        return '\n\n'.join(result)
        
    except Exception as e:
        return f"Extraction error: {str(e)}"

def detect_bubble_regions(img):
    """
    Detect chat bubble regions using color analysis
    """
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to find regions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 10)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubble_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter for bubble-like regions
        if (area > 1000 and  # Minimum area
            w > 50 and h > 20 and  # Minimum dimensions
            w < width * 0.8 and h < height * 0.5 and  # Maximum dimensions
            y > height * 0.05 and y < height * 0.95):  # Exclude edges
            
            # Determine if region is on left or right side
            center_x = x + w/2
            side = 'right' if center_x > width * 0.5 else 'left'
            
            bubble_regions.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'center_x': center_x,
                'center_y': y + h/2,
                'side': side
            })
    
    return bubble_regions

def determine_speaker(line_x, line_y, img_width, bubble_regions):
    """
    Determine speaker based on bubble regions or fallback to position
    """
    # First, try to match with detected bubble regions
    for bubble in bubble_regions:
        # Check if the line center is within this bubble region (with some tolerance)
        if (bubble['x'] - 20 <= line_x <= bubble['x'] + bubble['w'] + 20 and
            bubble['y'] - 20 <= line_y <= bubble['y'] + bubble['h'] + 20):
            
            # Use the bubble's side to determine speaker
            return "USER:" if bubble['side'] == 'right' else "Chatbot:"
    
    # Fallback: use simple position-based determination
    # More conservative threshold for right side (user messages)
    if line_x > img_width * 0.6:
        return "USER:"
    else:
        return "Chatbot:"

def group_words_into_lines(words_data, line_threshold=15):
    """
    Group words into lines based on Y position
    """
    if not words_data:
        return []
    
    # Sort words by Y position
    words_data.sort(key=lambda x: x['y'])
    
    lines = []
    current_line = {'words': [words_data[0]], 'y': words_data[0]['y']}
    
    for word in words_data[1:]:
        # If the word is on the same line (small Y difference)
        if abs(word['y'] - current_line['y']) <= line_threshold:
            current_line['words'].append(word)
        else:
            # New line
            if current_line['words']:
                lines.append(current_line)
            current_line = {'words': [word], 'y': word['y']}
    
    # Add the last line
    if current_line['words']:
        lines.append(current_line)
    
    # Sort words in each line by X position (left to right)
    for line in lines:
        line['words'].sort(key=lambda x: x['x'])
    
    return lines

def clean_extracted_text(text):
    """
    Clean and normalize extracted text
    """
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    
    # Remove timestamps (like 10:41, 12:30 PM)
    text = re.sub(r'\b\d{1,2}:\d{2}(?:\s?[AP]M)?\b', '', text)
    
    # Remove common OCR artifacts
    text = re.sub(r'[|\\/_@#$%^&*+=<>{}[\]]', '', text)
    
    # Remove usernames/mentions (like @username)
    text = re.sub(r'^@\w+\s*', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix punctuation spacing
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    return text.strip()

def save_results(results, output_file="extracted_chat_results.txt"):
    """
    Save all results to a text file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"Image: {result['filename']}\n")
            f.write(f"Path: {result['image_path']}\n")
            f.write("Extracted Text:\n")
            f.write(result['extracted_text'])
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"\nAll results saved to: {output_file}")

# Main execution
if __name__ == "__main__":
    print("Starting chat text extraction from downloaded images...")
    
    # Process all images in the folder
    results = process_downloaded_images("downloaded_images")
    
    if results:
        print(f"\nProcessed {len(results)} images successfully")
        
        # Save all results
        save_results(results)
        
        print("\nExtraction completed!")
    else:
        print("No images were processed")
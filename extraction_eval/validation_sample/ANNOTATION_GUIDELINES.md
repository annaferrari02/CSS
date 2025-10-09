# ANNOTATION GUIDELINES

## Overview
Manually transcribe text from chat screenshots to validate automated extraction accuracy.

## Instructions

### 1. Text Transcription
- Type EXACTLY what you see in each chat bubble
- Preserve capitalization, punctuation, spacing
- Include emojis: ❤️, 😊, etc.
- Use "..." if text is truncated
- Note illegible text in 'notes' column

### 2. Speaker Identification  
- **User**: Right side, lighter background
- **Chatbot**: Left side, darker background
- Use exactly: "User" or "Chatbot"

### 3. Turn Numbering
- Number from top to bottom: 1, 2, 3...
- Multiple consecutive bubbles from same speaker = same turn_id

### 4. Dialogue ID
- Create unique ID per screenshot: dial_0001, dial_0002, etc.

### 5. Notes Column
Standard tags:
- `low_quality`: Blurry/pixelated
- `truncated`: Text cut off
- `overlapping`: Bubbles overlap
- `ambiguous_speaker`: Can't determine speaker
- `contains_image`: Has image/gif, not just text

### 6. Annotator ID
Use consistent ID (e.g., "annotator_1")

## Example
```csv
screenshot_id,dialogue_id,turn_id,speaker,text,notes,annotator_id
sample_0001,dial_0001,1,User,"hey how are you",,annotator_1
sample_0001,dial_0001,2,Chatbot,"Great! How about you? 😊",,annotator_1
Quality Checks

Re-read for accuracy
Verify speaker labels
Check sequential numbering
Flag uncertainties

Contact: [your_email]

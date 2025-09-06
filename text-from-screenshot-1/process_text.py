
import pandas as pd
import re
import os

def parse_chat_file(file_path):
    """
    Parse a TXT file with chat logs and convert to CSV format.
    
    Args:
        file_path (str): Path to the input TXT file
    
    Returns:
        pandas.DataFrame: DataFrame with conversation_id, party, text columns
    """
    
    conversations = []
    conversation_counter = 1
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    
    # Split by the separator line
    sections = content.split('=' * 80)
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        
        # Find the "Extracted Text:" line
        extracted_start = -1
        for i, line in enumerate(lines):
            if "Extracted Text:" in line:
                extracted_start = i + 1
                break
        
        if extracted_start == -1:
            continue
        
        # Process the chat lines
        chat_lines = lines[extracted_start:]
        current_conversation_id = f"conv_{conversation_counter:03d}"
        
        # Track if we found any valid chat content
        found_valid_content = False
        
        for line in chat_lines:
            line = line.strip()
            if not line:
                continue
                
            # Match patterns like "USER: text" or "Chatbot: text"
            match = re.match(r'^(USER|Chatbot):\s*(.*)$', line)
            if match:
                party = match.group(1)
                text = match.group(2).strip()
                
                # Skip empty messages
                if text:
                    conversations.append({
                        'conversation_id': current_conversation_id,
                        'party': party,
                        'text': text
                    })
                    found_valid_content = True
        
        # Only increment conversation counter if we found valid content
        if found_valid_content:
            conversation_counter += 1
    
    return pd.DataFrame(conversations)

def clean_text(text):
    """
    Clean text by removing extra whitespace and fixing common OCR errors.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Fix common OCR errors (you can add more as needed)
    replacements = {
        ' .': '.',
        ' ,': ',',
        ' !': '!',
        ' ?': '?',
        ' ;': ';',
        ' :': ':',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def merge_fragmented_messages(df):
    """
    Merge consecutive messages from the same party in the same conversation.
    This handles cases where messages are split across multiple lines.
    
    Args:
        df (pandas.DataFrame): DataFrame with chat data
        
    Returns:
        pandas.DataFrame: DataFrame with merged messages
    """
    if df.empty:
        return df
    
    merged_conversations = []
    
    # Group by conversation_id
    for conv_id in df['conversation_id'].unique():
        conv_df = df[df['conversation_id'] == conv_id].copy()
        conv_df = conv_df.reset_index(drop=True)
        
        merged_messages = []
        current_party = None
        current_text = ""
        
        for _, row in conv_df.iterrows():
            if row['party'] == current_party:
                # Same party, merge the text
                current_text += " " + row['text']
            else:
                # Different party, save previous message and start new one
                if current_party is not None:
                    merged_messages.append({
                        'conversation_id': conv_id,
                        'party': current_party,
                        'text': clean_text(current_text)
                    })
                
                current_party = row['party']
                current_text = row['text']
        
        # Don't forget the last message
        if current_party is not None:
            merged_messages.append({
                'conversation_id': conv_id,
                'party': current_party,
                'text': clean_text(current_text)
            })
        
        merged_conversations.extend(merged_messages)
    
    return pd.DataFrame(merged_conversations)

def main():
    # Input and output file paths
    input_file = "C:/Users/anna2/OneDrive/Desktop/CSS/text-from-screenshot-1/extracted_chat_results.txt"  # Change this to your input file name
    output_file = "human_ai_chatlogs.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure the file exists and update the 'input_file' variable.")
        return
    
    print(f"Processing {input_file}...")
    
    try:
        # Parse the file
        df = parse_chat_file(input_file)
        
        if df.empty:
            print("No valid chat conversations found in the file!")
            return
        
        print(f"Found {len(df)} raw messages in {df['conversation_id'].nunique()} conversations")
        
        # Merge fragmented messages
        df_merged = merge_fragmented_messages(df)
        
        print(f"After merging: {len(df_merged)} messages in {df_merged['conversation_id'].nunique()} conversations")
        
        # Save to CSV
        df_merged.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"Successfully created {output_file}")
        print(f"Final stats:")
        print(f"  - Total messages: {len(df_merged)}")
        print(f"  - Total conversations: {df_merged['conversation_id'].nunique()}")
        print(f"  - USER messages: {len(df_merged[df_merged['party'] == 'USER'])}")
        print(f"  - Chatbot messages: {len(df_merged[df_merged['party'] == 'Chatbot'])}")
        
        # Show first few rows as preview
        print("\nPreview of first 10 rows:")
        print(df_merged.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
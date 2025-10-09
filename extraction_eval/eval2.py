import pandas as pd
import json
from pathlib import Path
import re
from collections import defaultdict
import random
import shutil
import glob
import sys
import traceback

print("="*70)
print("SCRIPT STARTED - DEBUG MODE")
print("="*70)
print(f"Python version: {sys.version}")
print(f"Current directory: {Path.cwd()}")
print()

class RedditScreenshotMapper:
    """
    Mappa screenshot Reddit (formato: postid_index_mediaid) ai metadata delle submissions
    """
    
    def __init__(self, 
                 screenshots_dir='../data/screenshots',
                 submissions_dir='../data',
                 output_dir='./validation_sample'):
        
        print(f"Initializing RedditScreenshotMapper...")
        print(f"  Screenshots dir: {screenshots_dir}")
        print(f"  Submissions dir: {submissions_dir}")
        print(f"  Output dir: {output_dir}")
        
        self.screenshots_dir = Path(screenshots_dir).resolve()
        self.submissions_dir = Path(submissions_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        
        print(f"\nResolved paths:")
        print(f"  Screenshots: {self.screenshots_dir}")
        print(f"  Submissions: {self.submissions_dir}")
        print(f"  Output: {self.output_dir}")
        
        # Verifica esistenza directory
        print(f"\nChecking directories:")
        print(f"  Screenshots exists: {self.screenshots_dir.exists()}")
        print(f"  Submissions exists: {self.submissions_dir.exists()}")
        
        if not self.screenshots_dir.exists():
            print(f"  ⚠️  Creating screenshots directory: {self.screenshots_dir}")
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.submissions_dir.exists():
            print(f"  ❌ Submissions directory does not exist!")
            raise FileNotFoundError(f"Directory not found: {self.submissions_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Output directory ready")
    
    def load_all_submissions(self):
        """
        Carica tutti i file JSON/JSONL dalla directory submissions
        """
        print(f"\n{'='*70}")
        print(f"LOADING SUBMISSIONS FROM: {self.submissions_dir}")
        print(f"{'='*70}")
        
        all_submissions = []
        
        # Pattern per trovare file JSON/JSONL
        print("\nSearching for JSON files...")
        
        json_files = []
        
        # Cerca nella directory principale
        for ext in ['*.json', '*.jsonl']:
            pattern = self.submissions_dir / ext
            found = list(self.submissions_dir.glob(ext))
            print(f"  Pattern {ext}: found {len(found)} files")
            json_files.extend(found)
        
        # Cerca nelle subdirectory
        for ext in ['**/*.json', '**/*.jsonl']:
            pattern = self.submissions_dir / ext
            found = list(self.submissions_dir.glob(ext))
            print(f"  Pattern {ext} (recursive): found {len(found)} files")
            json_files.extend(found)
        
        # Rimuovi duplicati
        json_files = list(set(json_files))
        
        print(f"\nTotal unique JSON/JSONL files found: {len(json_files)}")
        
        if len(json_files) == 0:
            print("\n❌ NO JSON FILES FOUND!")
            print(f"Contents of {self.submissions_dir}:")
            try:
                for item in sorted(self.submissions_dir.iterdir()):
                    print(f"  - {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
            except Exception as e:
                print(f"  Error listing directory: {e}")
            return []
        
        print(f"\nFiles to process:")
        for f in sorted(json_files):
            print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        
        # Carica ogni file
        for filepath in sorted(json_files):
            filepath = Path(filepath)
            print(f"\n{'─'*70}")
            print(f"Processing: {filepath.name}")
            print(f"Size: {filepath.stat().st_size / 1024:.1f} KB")
            
            try:
                # Prova JSONL (una riga per record)
                if filepath.suffix == '.jsonl' or self._is_jsonl(filepath):
                    print(f"  Format: JSONL (line-delimited)")
                    records = self.load_jsonl(filepath)
                    all_submissions.extend(records)
                    print(f"  ✓ Loaded {len(records)} records")
                
                # Altrimenti JSON standard
                else:
                    print(f"  Format: JSON")
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Se è una lista
                        if isinstance(data, list):
                            all_submissions.extend(data)
                            print(f"  ✓ Loaded {len(data)} records (array)")
                        
                        # Se è un dict con chiave 'data' o simili
                        elif isinstance(data, dict):
                            if 'data' in data and isinstance(data['data'], list):
                                all_submissions.extend(data['data'])
                                print(f"  ✓ Loaded {len(data['data'])} records (dict['data'])")
                            else:
                                # Singolo record
                                all_submissions.append(data)
                                print(f"  ✓ Loaded 1 record (single object)")
            
            except Exception as e:
                print(f"  ✗ Error: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue
        
        print(f"\n{'='*70}")
        print(f"TOTAL SUBMISSIONS LOADED: {len(all_submissions)}")
        print(f"{'='*70}")
        
        return all_submissions
    
    def _is_jsonl(self, filepath):
        """
        Controlla se un file è JSONL leggendo le prime righe
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
                
                # Se entrambe le righe sono JSON validi, probabilmente è JSONL
                if first_line and second_line:
                    json.loads(first_line)
                    json.loads(second_line)
                    return True
        except:
            pass
        
        return False
    
    def load_jsonl(self, filepath):
        """Carica file JSONL (una riga JSON per record)"""
        records = []
        errors = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # salta righe vuote
                    continue
                
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    errors += 1
                    if errors <= 3:  # mostra solo primi 3 errori
                        print(f"    Warning line {i}: {e}")
                    continue
        
        if errors > 3:
            print(f"    ... and {errors - 3} more errors")
        
        return records
    
    def parse_screenshot_filename(self, filepath):
        """
        Parse filename formato: {post_id}_{image_index}_{media_id}
        """
        filename = filepath.stem
        
        # Pattern: postid_index_mediaid
        match = re.match(r'^([a-z0-9]+)_(\d+)_([a-z0-9]+)$', filename, re.IGNORECASE)
        
        if match:
            return {
                'filename': filename,
                'filepath': str(filepath),
                'post_id': match.group(1),
                'image_index': int(match.group(2)),
                'media_id': match.group(3)
            }
        else:
            # Fallback: prova solo post_id
            match_simple = re.match(r'^([a-z0-9]+)', filename, re.IGNORECASE)
            if match_simple:
                return {
                    'filename': filename,
                    'filepath': str(filepath),
                    'post_id': match_simple.group(1),
                    'image_index': 0,
                    'media_id': None
                }
        
        return None
    
    def extract_submission_metadata(self, submission):
        """
        Estrae metadata rilevanti da un submission Reddit
        FIX: gestisce correttamente valori None
        """
        base_meta = {
            'post_id': submission.get('id'),
            'subreddit': submission.get('subreddit'),
            'author': submission.get('author'),
            'title': submission.get('title'),
            'created_utc': submission.get('created_utc'),
            'score': submission.get('score'),
            'num_comments': submission.get('num_comments'),
            'permalink': submission.get('permalink'),
            'url': submission.get('url'),
            'is_gallery': submission.get('is_gallery', False),
        }
        
        images = []
        
        # FIX: controlla che gallery_data non sia None E sia un dict
        gallery_data = submission.get('gallery_data')
        if submission.get('is_gallery') and gallery_data and isinstance(gallery_data, dict):
            gallery_items = gallery_data.get('items', [])
            media_metadata = submission.get('media_metadata', {})
            
            if isinstance(media_metadata, dict):
                for idx, item in enumerate(gallery_items):
                    if not isinstance(item, dict):
                        continue
                    
                    media_id = item.get('media_id')
                    
                    if media_id and media_id in media_metadata:
                        media_info = media_metadata[media_id]
                        
                        if not isinstance(media_info, dict):
                            continue
                        
                        # Prendi URL più alta risoluzione
                        image_url = None
                        if 's' in media_info and isinstance(media_info['s'], dict):
                            image_url = media_info['s'].get('u')
                        
                        images.append({
                            'image_index': idx,
                            'media_id': media_id,
                            'image_url': image_url,
                            'mime_type': media_info.get('m'),
                            'status': media_info.get('status')
                        })
        
        # Post con singola immagine (non gallery)
        elif submission.get('url'):
            url = submission.get('url')
            if url and isinstance(url, str):
                if any(ext in url.lower() for ext in ['.jpg', '.png', '.jpeg', '.gif', '.webp']):
                    images.append({
                        'image_index': 0,
                        'media_id': None,
                        'image_url': url,
                        'mime_type': None,
                        'status': 'valid'
                    })
        
        base_meta['n_images'] = len(images)
        
        return base_meta, images

    def create_mapping_dataframe(self):
        """
        Crea DataFrame con mapping screenshot -> metadata
        """
        print("\n" + "="*70)
        print("CREATING SCREENSHOT-METADATA MAPPING")
        print("="*70)
        
        # 1. Carica submissions
        submissions = self.load_all_submissions()
        
        if len(submissions) == 0:
            print("\n❌ No submissions loaded!")
            return pd.DataFrame(), []
        
        # 2. Crea indice
        print(f"\nCreating post index...")
        post_index = {}
        posts_with_images = 0
        
        for i, sub in enumerate(submissions):
            if i % 1000 == 0:
                print(f"  Processing submission {i}/{len(submissions)}")
            
            base_meta, images = self.extract_submission_metadata(sub)
            post_id = base_meta['post_id']
            
            if post_id and len(images) > 0:
                post_index[post_id] = {
                    'metadata': base_meta,
                    'images': images
                }
                posts_with_images += 1
        
        print(f"\n✓ Indexed {len(post_index)} submissions with images")
        print(f"  (out of {len(submissions)} total submissions)")
        print(f"  Posts with images: {posts_with_images}")
        
        # 3. Scansiona screenshot
        print(f"\nScanning screenshot directory: {self.screenshots_dir}")
        print("  Searching in subdirectories...")
        screenshot_files = list(self.screenshots_dir.glob('**/*.png')) + \
                        list(self.screenshots_dir.glob('**/*.jpg')) + \
                        list(self.screenshots_dir.glob('**/*.jpeg'))
        
        print(f"Found {len(screenshot_files)} screenshot files")
        
        if len(screenshot_files) == 0:
            print(f"\n❌ NO SCREENSHOTS FOUND in {self.screenshots_dir}")
            print(f"Contents of directory:")
            try:
                for item in list(self.screenshots_dir.iterdir())[:10]:
                    print(f"  - {item.name}")
            except Exception as e:
                print(f"  Error listing: {e}")
            return pd.DataFrame(), []
        
        # Sample di nomi file
        print(f"\nSample screenshot filenames:")
        for f in screenshot_files[:5]:
            print(f"  - {f.name}")
        
        # 4. Match
        print(f"\nMatching screenshots to metadata...")
        mappings = []
        unmatched = []
        
        for i, filepath in enumerate(screenshot_files):
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{len(screenshot_files)} screenshots")
            
            parsed = self.parse_screenshot_filename(filepath)
            
            if not parsed:
                unmatched.append(str(filepath))
                continue
            
            post_id = parsed['post_id']
            image_index = parsed['image_index']
            media_id = parsed['media_id']
            
            if post_id in post_index:
                post_data = post_index[post_id]
                base_meta = post_data['metadata']
                images = post_data['images']
                
                matched_image = None
                if image_index < len(images):
                    candidate = images[image_index]
                    
                    if media_id and candidate['media_id']:
                        if candidate['media_id'] == media_id:
                            matched_image = candidate
                            match_confidence = 1.0
                        else:
                            matched_image = candidate
                            match_confidence = 0.7
                    else:
                        matched_image = candidate
                        match_confidence = 0.9
                else:
                    if len(images) == 1:
                        matched_image = images[0]
                        match_confidence = 0.6
                
                if matched_image:
                    mappings.append({
                        **parsed,
                        **base_meta,
                        'image_url': matched_image.get('image_url'),
                        'media_id_matched': matched_image.get('media_id'),
                        'match_confidence': match_confidence
                    })
                else:
                    unmatched.append(str(filepath))
            else:
                unmatched.append(str(filepath))
        
        df = pd.DataFrame(mappings)
        
        print(f"\n{'='*70}")
        print("MAPPING RESULTS")
        print(f"{'='*70}")
        print(f"Total screenshots: {len(screenshot_files)}")
        print(f"Successfully matched: {len(mappings)} ({len(mappings)/len(screenshot_files)*100:.1f}%)")
        print(f"Unmatched: {len(unmatched)} ({len(unmatched)/len(screenshot_files)*100:.1f}%)")
        
        if len(df) > 0:
            print(f"\nMatch confidence distribution:")
            for conf, count in df['match_confidence'].value_counts().sort_index(ascending=False).items():
                print(f"  {conf:.1f}: {count} ({count/len(df)*100:.1f}%)")
            
            print(f"\nSubreddits:")
            for sub, count in df['subreddit'].value_counts().head(10).items():
                print(f"  {sub}: {count}")
        
        if unmatched:
            print(f"\nSample unmatched ({len(unmatched)} total):")
            for f in unmatched[:5]:
                print(f"  - {Path(f).name}")
        
        return df, unmatched
    
    def save_mapping(self, df, output_path=None):
        """Salva mapping"""
        if output_path is None:
            output_path = self.output_dir / 'screenshots_mapping.csv'
        
        df.to_csv(output_path, index=False)
        print(f"\n✅ Mapping saved to: {output_path}")
        
        return output_path


# [ScreenshotSampler class identica a prima - omessa per brevità]
class ScreenshotSampler:
    """
    Campionamento stratificato dagli screenshot mappati
    """
    
    def __init__(self, 
                 mapping_df,
                 output_dir='./validation_sample',
                 sample_size=90,
                 min_confidence=0.7,
                 random_seed=42):
        
        self.mapping_df = mapping_df
        self.output_dir = Path(output_dir)
        self.sample_size = sample_size
        self.min_confidence = min_confidence
        self.random_seed = random_seed
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(self.random_seed)
    
    def filter_by_confidence(self):
        """Filtra solo match ad alta confidenza"""
        filtered = self.mapping_df[
            self.mapping_df['match_confidence'] >= self.min_confidence
        ].copy()
        
        print(f"\nFiltered to {len(filtered)} screenshots with confidence ≥ {self.min_confidence}")
        return filtered
    
    def stratified_sample(self, df):
        """Campionamento stratificato per subreddit"""
        print(f"\n{'='*70}")
        print(f"STRATIFIED SAMPLING (target: {self.sample_size})")
        print(f"{'='*70}")
        
        if 'subreddit' not in df.columns:
            print("No subreddit column, using random sampling")
            return df.sample(n=min(self.sample_size, len(df)), random_state=self.random_seed)
        
        subreddit_counts = df['subreddit'].value_counts()
        print(f"\nOriginal distribution:")
        for sub, count in subreddit_counts.items():
            print(f"  {sub}: {count}")
        
        samples = []
        remaining = self.sample_size
        
        for subreddit, count in subreddit_counts.items():
            proportion = count / len(df)
            n_samples = max(10, int(self.sample_size * proportion))
            n_samples = min(n_samples, count, remaining)
            
            if n_samples > 0:
                subreddit_df = df[df['subreddit'] == subreddit]
                sampled = subreddit_df.sample(n=n_samples, random_state=self.random_seed)
                samples.append(sampled)
                remaining -= n_samples
                
                print(f"  Sampled {n_samples} from {subreddit}")
        
        sample_df = pd.concat(samples, ignore_index=True)
        
        if len(sample_df) > self.sample_size:
            sample_df = sample_df.sample(n=self.sample_size, random_state=self.random_seed)
        
        print(f"\nFinal sample size: {len(sample_df)}")
        return sample_df
    
    def copy_screenshots(self, sample_df):
        """Copia screenshot nella cartella validation"""
        screenshot_dir = self.output_dir / 'screenshots'
        screenshot_dir.mkdir(exist_ok=True)
        
        print(f"\nCopying {len(sample_df)} screenshots...")
        
        copied = 0
        for idx, row in sample_df.iterrows():
            src = Path(row['filepath'])
            
            if src.exists():
                # Nome più descrittivo
                dst_name = f"sample_{idx:04d}_{row['post_id']}_{row['image_index']}{src.suffix}"
                dst = screenshot_dir / dst_name
                
                shutil.copy2(src, dst)
                sample_df.at[idx, 'sample_path'] = str(dst.relative_to(self.output_dir))
                copied += 1
            else:
                print(f"⚠️  File not found: {src}")
        
        print(f"✅ Copied {copied}/{len(sample_df)} files")
        return sample_df
    
    def create_annotation_template(self, sample_df):
        """Crea template per annotazione manuale"""
        template = []
        
        for idx, row in sample_df.iterrows():
            template.append({
                'screenshot_id': f"sample_{idx:04d}",
                'screenshot_path': row.get('sample_path', ''),
                'post_id': row['post_id'],
                'subreddit': row['subreddit'],
                'image_index': row['image_index'],
                'match_confidence': row['match_confidence'],
                'post_title': row.get('title', ''),
                'post_url': row.get('url', ''),
                'dialogue_id': '',  # Da compilare
                'turn_id': '',      # Da compilare
                'speaker': '',      # Da compilare (User/Chatbot)
                'text': '',         # Da compilare
                'notes': '',        # Da compilare
                'annotator_id': ''  # Da compilare
            })
        
        template_df = pd.DataFrame(template)
        
        # Salva CSV
        csv_path = self.output_dir / 'annotation_template.csv'
        template_df.to_csv(csv_path, index=False)
        
        # Salva Excel (più user-friendly)
        excel_path = self.output_dir / 'annotation_template.xlsx'
        template_df.to_excel(excel_path, index=False, sheet_name='Annotations')
        
        print(f"\n✅ Annotation templates created:")
        print(f"   - CSV: {csv_path}")
        print(f"   - Excel: {excel_path}")
        
        return template_df
    
    def create_guidelines(self):
        """Crea annotation guidelines"""
        guidelines = """# ANNOTATION GUIDELINES

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
"""
        guidelines_path = self.output_dir / 'ANNOTATION_GUIDELINES.md'
        with open(guidelines_path, 'w', encoding='utf-8') as f:
            f.write(guidelines)
        
        print(f"✅ Guidelines saved to: {guidelines_path}")

    def generate_report(self, sample_df):
        """Report finale del sampling"""
        report = f"""
        {'='*70}
        VALIDATION SAMPLE REPORT
        {'='*70}
        SAMPLE STATISTICS
        Total screenshots: {len(sample_df)}
        Random seed: {self.random_seed}
        Min confidence threshold: {self.min_confidence}
        Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
        SUBREDDIT DISTRIBUTION
        {sample_df['subreddit'].value_counts().to_string()}
        MATCH CONFIDENCE
        Mean: {sample_df['match_confidence'].mean():.3f}
        Median: {sample_df['match_confidence'].median():.3f}
        Min: {sample_df['match_confidence'].min():.3f}
        METADATA AVAILABLE
        Posts represented: {sample_df['post_id'].nunique()}
        Average images per post: {sample_df.groupby('post_id').size().mean():.1f}
        FILES GENERATED

        screenshots/ ({len(sample_df)} files)
        annotation_template.csv
        annotation_template.xlsx
        ANNOTATION_GUIDELINES.md
        sample_metadata.csv
        sampling_report.txt

        ESTIMATED ANNOTATION TIME
        Screenshots per annotator: {len(sample_df)}
        Average time: 2-3 min/screenshot
        Total estimated: {len(sample_df) * 2.5 / 60:.1f} - {len(sample_df) * 3.5 / 60:.1f} hours
        NEXT STEPS

        Review screenshots in screenshots/ folder
        Distribute annotation_template.xlsx to annotators
        Complete annotations independently
        Calculate inter-annotator agreement
        Run validation analysis
        """
        report_path = self.output_dir / 'sampling_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print("\n" + report)
        print(f"✅ Report saved to: {report_path}")

    def run(self):
        """Pipeline completa di sampling"""
        print("\n" + "="*70)
        print("SCREENSHOT SAMPLING PIPELINE")
        print("="*70)
        # 1. Filtra per confidenza
        filtered_df = self.filter_by_confidence()
        
        if len(filtered_df) < self.sample_size:
            print(f"\n⚠️  WARNING: Only {len(filtered_df)} screenshots meet confidence threshold")
            print(f"   Consider lowering min_confidence (currently {self.min_confidence})")
            self.sample_size = len(filtered_df)
        
        # 2. Campionamento stratificato
        sample_df = self.stratified_sample(filtered_df)
        
        # 3. Salva metadata del sample
        sample_meta_path = self.output_dir / 'sample_metadata.csv'
        sample_df.to_csv(sample_meta_path, index=False)
        print(f"✅ Sample metadata saved to: {sample_meta_path}")
        
        # 4. Copia screenshot
        sample_df = self.copy_screenshots(sample_df)
        
        # 5. Crea template annotazione
        self.create_annotation_template(sample_df)
        
        # 6. Crea guidelines
        self.create_guidelines()
        
        # 7. Report finale
        self.generate_report(sample_df)
        
        print(f"\n{'='*70}")
        print("✅ SAMPLING COMPLETED SUCCESSFULLY!")
        print(f"📁 All files in: {self.output_dir.absolute()}")
        print(f"{'='*70}\n")
        
        return sample_df






if __name__ == "__main__":
    try:
        print("="*70)
        print("REDDIT SCREENSHOT VALIDATION SAMPLER")
        print("="*70)
        
        screenshots_dir = '../data/screenshots'
        submissions_dir = '../data'
        output_dir = './validation_sample'
        
        print(f"\nConfiguration:")
        print(f"  Screenshots: {screenshots_dir}")
        print(f"  Submissions: {submissions_dir}")
        print(f"  Output: {output_dir}")
        
        # STEP 1: Mapping
        print("\n" + "="*70)
        print("STEP 1: MAPPING SCREENSHOTS TO METADATA")
        print("="*70)
        
        mapper = RedditScreenshotMapper(
            screenshots_dir=screenshots_dir,
            submissions_dir=submissions_dir,
            output_dir=output_dir
        )
        
        mapping_df, unmatched = mapper.create_mapping_dataframe()
        
        if len(mapping_df) == 0:
            print("\n❌ ERROR: No mappings created!")
            sys.exit(1)
        
        mapper.save_mapping(mapping_df)
        
        # STEP 2: Sampling
        print("\n" + "="*70)
        print("STEP 2: STRATIFIED SAMPLING")
        print("="*70)
        
        sampler = ScreenshotSampler(
            mapping_df=mapping_df,
            output_dir=output_dir,
            sample_size=90,
            min_confidence=0.7,
            random_seed=42
        )
        
        sample = sampler.run()
        
        print("\n🎉 All done! Next steps:")
        print("   1. cd validation_sample")
        print("   2. Review screenshots/ folder")
        print("   3. Open annotation_template.xlsx")
        print("   4. Follow ANNOTATION_GUIDELINES.md")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
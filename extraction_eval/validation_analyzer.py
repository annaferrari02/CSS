import pandas as pd
import numpy as np
from pathlib import Path
import Levenshtein
from collections import defaultdict
import json
import re

class ValidationAnalyzer:
    """
    Confronta annotazioni manuali (gold standard) con estrazione Pytesseract
    """
    
    def __init__(self, 
                 gold_standard_path='./validation_sample/gold_standard_annotations.csv',
                 pytesseract_path='../data/human-ai-chatlogs.csv',
                 screenshot_mapping_path='./validation_sample/screenshots_mapping.csv'):
        """
        Args:
            gold_standard_path: CSV con annotazioni manuali
            pytesseract_path: CSV con estrazione Pytesseract
            screenshot_mapping_path: Mapping screenshot -> post_id
        """
        print("="*70)
        print("VALIDATION ANALYZER - PYTESSERACT VS MANUAL")
        print("="*70)
        
        # 1. Carica mapping screenshot -> conv_id
        print(f"\nLoading screenshot mapping...")
        self.mapping = pd.read_csv(screenshot_mapping_path)
        print(f"  ✓ {len(self.mapping)} screenshots mapped")
        
        # Crea dizionario screenshot_id -> post_id (che è il conv_id)
        self.screenshot_to_conv = dict(zip(
            self.mapping['screenshot_id'],
            self.mapping['post_id']
        ))
        
        # 2. Carica annotazioni manuali
        print(f"\nLoading gold standard: {gold_standard_path}")
        self.gold = pd.read_csv(gold_standard_path)
        
        # Aggiungi conv_id alle annotazioni manuali
        self.gold['conv_id'] = self.gold['screenshot_id'].map(self.screenshot_to_conv)
        
        print(f"  ✓ Loaded {len(self.gold)} manual annotations")
        print(f"  Screenshots: {self.gold['screenshot_id'].nunique()}")
        print(f"  Conversations: {self.gold['conv_id'].nunique()}")
        
        # 3. Carica estrazione Pytesseract
        print(f"\nLoading Pytesseract extraction: {pytesseract_path}")
        
        # Carica CSV senza header (basandosi sul tuo formato)
        self.auto = pd.read_csv(
            pytesseract_path,
            names=['conversation_id', 'party', 'text'],
            header=None
        )
        
        print(f"  ✓ Loaded {len(self.auto)} automated annotations")
        print(f"  Conversations: {self.auto['conv_id'].nunique()}")
        
        # 4. Filtra Pytesseract solo per conv_id che abbiamo nel gold standard
        gold_conv_ids = set(self.gold['conversation_id'].dropna().unique())
        self.auto = self.auto[self.auto['conversation_id'].isin(gold_conv_ids)].copy()
        
        print(f"  ✓ Filtered to {len(self.auto)} annotations matching gold standard")
        
        # 5. Aggiungi turn_id sequenziale per Pytesseract
        self.auto['turn_id'] = self.auto.groupby('conversation_id').cumcount() + 1
        
        # Standardizza
        self._standardize_data()
    
    def _standardize_data(self):
        """Standardizza party names e formati"""
        # Standardizza party (case-insensitive)
        self.gold['party'] = self.gold['party'].str.strip().str.title()
        self.auto['party'] = self.auto['party'].str.strip().str.title()
        
        # Mappa varianti comuni
        party_map = {
            'User': 'User',
            'USER': 'User',
            'user': 'User',
            'Chatbot': 'Chatbot',
            'CHATBOT': 'Chatbot',
            'chatbot': 'Chatbot',
            'Bot': 'Chatbot',
            'Assistant': 'Chatbot'
        }
        
        self.gold['party'] = self.gold['party'].replace(party_map)
        self.auto['party'] = self.auto['party'].replace(party_map)
    
    def normalize_text(self, text):
        """Normalizza testo per confronto fair"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip().lower()
        
        # Rimuovi spazi multipli
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def calculate_cer(self, gold_text, auto_text):
        """Character Error Rate usando Levenshtein distance"""
        gold_norm = self.normalize_text(gold_text)
        auto_norm = self.normalize_text(auto_text)
        
        if len(gold_norm) == 0:
            return 0.0 if len(auto_norm) == 0 else 1.0
        
        distance = Levenshtein.distance(gold_norm, auto_norm)
        return distance / len(gold_norm)
    
    def calculate_wer(self, gold_text, auto_text):
        """Word Error Rate"""
        gold_words = self.normalize_text(gold_text).split()
        auto_words = self.normalize_text(auto_text).split()
        
        if len(gold_words) == 0:
            return 0.0 if len(auto_words) == 0 else 1.0
        
        distance = Levenshtein.distance(' '.join(gold_words), ' '.join(auto_words))
        return distance / len(gold_words)
    
    def align_turns(self, conv_id):
        """
        Allinea turni per una conversazione specifica
        
        Usa Dynamic Time Warping semplificato basato su:
        - Ordine sequenziale
        - Speaker alternanza
        """
        gold_turns = self.gold[self.gold['conv_id'] == conv_id].sort_values('turn_id')
        auto_turns = self.auto[self.auto['conv_id'] == conv_id].sort_values('turn_id')
        
        alignments = []
        
        # Alignment semplice: allinea per posizione sequenziale
        max_turns = max(len(gold_turns), len(auto_turns))
        
        for i in range(max_turns):
            gold_turn = gold_turns.iloc[i] if i < len(gold_turns) else None
            auto_turn = auto_turns.iloc[i] if i < len(auto_turns) else None
            
            alignments.append({
                'gold': gold_turn,
                'auto': auto_turn,
                'position': i + 1
            })
        
        return alignments
    
    def calculate_metrics_per_conversation(self):
        """Calcola metriche per ogni conversazione"""
        print("\n" + "="*70)
        print("CALCULATING PER-CONVERSATION METRICS")
        print("="*70)
        
        results = []
        
        conv_ids = self.gold['conv_id'].dropna().unique()
        
        for conv_id in conv_ids:
            gold_conv = self.gold[self.gold['conv_id'] == conv_id].sort_values('turn_id')
            auto_conv = self.auto[self.auto['conv_id'] == conv_id].sort_values('turn_id')
            
            # Screenshot ID (per riferimento)
            screenshot_id = gold_conv['screenshot_id'].iloc[0]
            
            # 1. Turn Detection
            gold_n_turns = len(gold_conv)
            auto_n_turns = len(auto_conv)
            turn_detection_correct = (gold_n_turns == auto_n_turns)
            
            # 2. Allinea turni e calcola metriche
            alignments = self.align_turns(conv_id)
            
            cer_scores = []
            wer_scores = []
            party_correct = 0
            party_total = 0
            
            for align in alignments:
                gold_turn = align['gold']
                auto_turn = align['auto']
                
                # Skip se uno dei due manca
                if gold_turn is None or auto_turn is None:
                    continue
                
                # CER/WER
                cer = self.calculate_cer(gold_turn['text'], auto_turn['text'])
                wer = self.calculate_wer(gold_turn['text'], auto_turn['text'])
                
                cer_scores.append(cer)
                wer_scores.append(wer)
                
                # Speaker accuracy
                if gold_turn['party'] == auto_turn['party']:
                    party_correct += 1
                party_total += 1
            
            # Aggregate metrics
            results.append({
                'conv_id': conv_id,
                'screenshot_id': screenshot_id,
                'gold_n_turns': gold_n_turns,
                'auto_n_turns': auto_n_turns,
                'turn_detection_correct': turn_detection_correct,
                'mean_cer': np.mean(cer_scores) if cer_scores else np.nan,
                'median_cer': np.median(cer_scores) if cer_scores else np.nan,
                'mean_wer': np.mean(wer_scores) if wer_scores else np.nan,
                'median_wer': np.median(wer_scores) if wer_scores else np.nan,
                'party_accuracy': party_correct / party_total if party_total > 0 else np.nan,
                'matched_turns': len(cer_scores)
            })
        
        return pd.DataFrame(results)
    
    def calculate_overall_metrics(self, conv_results):
        """Calcola metriche aggregate"""
        print("\n" + "="*70)
        print("OVERALL METRICS")
        print("="*70)
        
        metrics = {
            # Turn Detection
            'Turn Detection Accuracy (%)': conv_results['turn_detection_correct'].mean() * 100,
            
            # Text Accuracy
            'Mean CER (%)': conv_results['mean_cer'].mean() * 100,
            'Median CER (%)': conv_results['median_cer'].median() * 100,
            'Mean WER (%)': conv_results['mean_wer'].mean() * 100,
            'Median WER (%)': conv_results['median_wer'].median() * 100,
            
            # Speaker Accuracy
            'Speaker Attribution Accuracy (%)': conv_results['party_accuracy'].mean() * 100,
            
            # Coverage
            'Conversations Evaluated': len(conv_results),
            'Total Gold Turns': conv_results['gold_n_turns'].sum(),
            'Total Auto Turns': conv_results['auto_n_turns'].sum(),
            'Matched Turns': conv_results['matched_turns'].sum()
        }
        
        return pd.Series(metrics)
    
    def analyze_error_patterns(self, conv_results):
        """Analizza pattern di errori comuni"""
        print("\n" + "="*70)
        print("ERROR PATTERN ANALYSIS")
        print("="*70)
        
        # 1. Conversazioni con alta error rate
        high_cer = conv_results[conv_results['mean_cer'] > 0.20]
        
        print(f"\nConversations with CER > 20%: {len(high_cer)} ({len(high_cer)/len(conv_results)*100:.1f}%)")
        
        if len(high_cer) > 0:
            print(f"  Mean CER in high-error cases: {high_cer['mean_cer'].mean()*100:.1f}%")
        
        # 2. Turn detection errors
        turn_errors = conv_results[~conv_results['turn_detection_correct']]
        
        print(f"\nTurn detection errors: {len(turn_errors)} ({len(turn_errors)/len(conv_results)*100:.1f}%)")
        
        if len(turn_errors) > 0:
            print(f"  Mean difference in turn count: {(turn_errors['auto_n_turns'] - turn_errors['gold_n_turns']).abs().mean():.1f} turns")
        
        # 3. Speaker attribution errors
        low_party = conv_results[conv_results['party_accuracy'] < 0.8]
        
        print(f"\nLow party accuracy (<80%): {len(low_party)} ({len(low_party)/len(conv_results)*100:.1f}%)")
        
        return {
            'high_cer_conversations': high_cer,
            'turn_detection_errors': turn_errors,
            'low_party_accuracy': low_party
        }
    
    def generate_detailed_report(self, conv_results, overall_metrics, error_analysis, output_dir='./validation_sample'):
        """Genera report dettagliato"""
        output_path = Path(output_dir)
        
        print("\n" + "="*70)
        print("GENERATING DETAILED REPORT")
        print("="*70)
        
        # 1. Salva metriche per conversazione
        conv_results_path = output_path / 'validation_per_conversation.csv'
        conv_results.to_csv(conv_results_path, index=False)
        print(f"\n✓ Per-conversation metrics: {conv_results_path}")
        
        # 2. Salva metriche overall
        overall_path = output_path / 'validation_overall_metrics.csv'
        overall_metrics.to_csv(overall_path)
        print(f"✓ Overall metrics: {overall_path}")
        
        # 3. Report testuale
        report = f"""
{'='*70}
VALIDATION REPORT - PYTESSERACT EXTRACTION ACCURACY
{'='*70}

DATASET
-------
Gold Standard Annotations: {len(self.gold)} turns across {self.gold['conv_id'].nunique()} conversations
Pytesseract Extraction: {len(self.auto)} turns across {self.auto['conv_id'].nunique()} conversations

OVERALL METRICS
---------------
{overall_metrics.to_string()}

ERROR ANALYSIS
--------------
High CER Conversations (>20%): {len(error_analysis['high_cer_conversations'])} ({len(error_analysis['high_cer_conversations'])/len(conv_results)*100:.1f}%)
Turn Detection Errors: {len(error_analysis['turn_detection_errors'])} ({len(error_analysis['turn_detection_errors'])/len(conv_results)*100:.1f}%)
Low Speaker Accuracy (<80%): {len(error_analysis['low_party_accuracy'])} ({len(error_analysis['low_party_accuracy'])/len(conv_results)*100:.1f}%)

CER DISTRIBUTION
----------------
Min: {conv_results['mean_cer'].min()*100:.2f}%
Q1:  {conv_results['mean_cer'].quantile(0.25)*100:.2f}%
Median: {conv_results['median_cer'].median()*100:.2f}%
Q3:  {conv_results['mean_cer'].quantile(0.75)*100:.2f}%
Max: {conv_results['mean_cer'].max()*100:.2f}%

INTERPRETATION
--------------
- CER < 10%: Excellent extraction quality
- CER 10-20%: Good extraction quality
- CER 20-30%: Moderate extraction quality
- CER > 30%: Poor extraction quality

Current Mean CER: {overall_metrics['Mean CER (%)']:.2f}%
Quality Assessment: {"Excellent" if overall_metrics['Mean CER (%)'] < 10 else "Good" if overall_metrics['Mean CER (%)'] < 20 else "Moderate" if overall_metrics['Mean CER (%)'] < 30 else "Poor"}

RECOMMENDATIONS
---------------
"""
        
        if overall_metrics['Mean CER (%)'] > 20:
            report += "- Consider improving OCR preprocessing (image quality, contrast)\n"
            report += "- Review cases with high CER for common error patterns\n"
        
        if overall_metrics['Turn Detection Accuracy (%)'] < 90:
            report += "- Improve turn boundary detection algorithm\n"
        
        if overall_metrics['Speaker Attribution Accuracy (%)'] < 90:
            report += "- Enhance party identification (position/color heuristics)\n"
        
        report += f"\n{'='*70}\n"
        
        report_path = output_path / 'validation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Detailed report: {report_path}")
        
        # 4. Esempi di errori
        if len(error_analysis['high_cer_conversations']) > 0:
            examples_path = output_path / 'high_error_examples.csv'
            error_analysis['high_cer_conversations'].to_csv(examples_path, index=False)
            print(f"✓ High-error examples: {examples_path}")
        
        print(f"\n{'='*70}")
        print("REPORT GENERATION COMPLETE")
        print(f"{'='*70}")
    
    def run_full_analysis(self):
        """Esegue analisi completa"""
        print("\n" + "="*70)
        print("STARTING FULL VALIDATION ANALYSIS")
        print("="*70)
        
        # 1. Metriche per conversazione
        conv_results = self.calculate_metrics_per_conversation()
        
        # 2. Metriche overall
        overall_metrics = self.calculate_overall_metrics(conv_results)
        
        print(f"\n{overall_metrics}")
        
        # 3. Analisi errori
        error_analysis = self.analyze_error_patterns(conv_results)
        
        # 4. Report
        self.generate_detailed_report(conv_results, overall_metrics, error_analysis)
        
        print("\n✅ VALIDATION ANALYSIS COMPLETE!")
        
        return {
            'per_conversation': conv_results,
            'overall': overall_metrics,
            'errors': error_analysis
        }


if __name__ == "__main__":
    try:
        analyzer = ValidationAnalyzer(
            gold_standard_path='./validation_sample/gold_standard_annotations.csv',
            pytesseract_path='../data/human-ai-chatlogs.csv',
            screenshot_mapping_path='./validation_sample/screenshots_mapping.csv'
        )
        
        results = analyzer.run_full_analysis()
        
        print("\n🎉 Analysis complete! Check validation_sample/ for results.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
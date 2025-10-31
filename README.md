
# CSS - Computational Social Science: Human-AI Chatbot Interaction Analysis

Analysis of emotional dependence and communication patterns in human-AI chatbot interactions.

## Overview

This repository contains code and analysis tools for investigating emotional dynamics, communication patterns, and behavioral characteristics in human-AI chatbot interactions. The project examines large-scale conversational data to understand how users engage with AI companions and the psychological dimensions of these interactions.

## Research Focus

The project explores:
- **Semantic and Topic alignment and adaptation** between users and AI chatbots
- **Communication styles** and linguistic patterns in human-AI conversations
- **Topic modeling and thematic analysis** of conversational content

## Key Analyses

### 1. Linguistic Analysis
- TF-IDF analysis for distinctive vocabulary identification
- Sentiment and emotion analysis using multiple frameworks

### 2. Topic Modeling
- BERTopic implementation for discovering latent conversation themes
- UMAP dimensionality reduction for visualization
- K-means clustering for topic categorization

### 3. NSFW Content Analysis
- DistilBERT-based NSFW classifier for binary content labeling
- NLTK text preprocessing with normalization and lemmatization for input optimization
- Message-level analysis for escalation patterns, temporal persistence metrics, and conversational contamination quantification

## Prerequisites

- **Python**: 3.8 or higher
- **Git**: (optional, for cloning)
- **Tesseract OCR**: Required for dialogue extraction from images

## Installation

### 1. Clone the Repository

```bash
# Navigate to your desired directory
cd C:\Users\YourUsername\OneDrive\Desktop

# Clone the repository
git clone https://github.com/annaferrari02/CSS.git
cd CSS
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Run installer (default path: `C:\Program Files\Tesseract-OCR`)
- Add to system PATH if needed

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

## Data

This project analyzes conversational data from various sources:
- **Reddit data**: Posts and comments from AI companion subreddits (r/Replika, r/CharacterAI, r/ChaiApp, etc.)
- **Chatbot dialogue corpus**: User-uploaded conversation screenshots
- **Survey data**: User characteristics and perceptions

**Data Collection Period**: January 2022 - December 2023

### Privacy and Ethics
- All data is anonymized and personally identifiable information removed
- Sensitive information is filtered prior to analysis

## Project Structure

```
CSS/
├── data/                   # Data files (not included in repo)
├── data_cleaned/           # Processed data files (.csv)
├── analysis/               # Jupyter notebooks and python scripts for analysis
├── preprocessing/         # Python scripts for data processing
├── output/                # Output files
├── latex source/          # Latex source of research report
├── requirements.txt       # Python dependencies
└── README.md              # This file
```



## Research Outputs

This repository supports research investigating:
1. How AI chatbots adapt to user emotional states
2. Linguistic divergence between human and AI communication styles
3. Psychosocial characteristics of users engaging with AI companions
4. Community dynamics and social positioning of AI companionship forums
5. Long-term relationship development with conversational AI


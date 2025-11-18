# NLP Text Preprocessing Pipeline

A comprehensive Natural Language Processing project that demonstrates fundamental text preprocessing techniques including data collection, cleaning, tokenization, encoding, padding, and visualization. This project implements a complete preprocessing pipeline from raw text to model-ready datasets.

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Components](#-project-components)
- [Technologies Used](#️-technologies-used)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Dataset Information](#-dataset-information)
- [Key Concepts](#-key-concepts-demonstrated)
- [Visualizations](#-visualizations)
- [Learning Outcomes](#-learning-outcomes)
- [Project Structure](#-project-structure)
- [Customization](#-customization)
- [Future Enhancements](#-future-enhancements)
- [References](#-references)

## Overview

This project provides hands-on experience with core NLP preprocessing tasks using Mark Twain's "The Adventures of Huckleberry Finn" as the primary dataset. The pipeline transforms raw text into structured, encoded data suitable for machine learning models, covering everything from basic text cleaning to creating PyTorch datasets.

## Key Features

- **Text Data Collection**: Download and load text datasets from URLs
- **Advanced Text Cleaning**: Preserves contractions while normalizing punctuation
- **Sentence Tokenization**: Uses NLTK's sentence tokenizer for accurate splitting
- **Word Tokenization**: Splits text into individual tokens/words
- **Vocabulary Building**: Creates word-to-integer mappings with `<UNK>` and `<PAD>` tokens
- **Sequence Encoding**: Converts text tokens to numerical representations
- **Padding & Truncation**: Handles variable-length sequences
- **Data Visualization**: Word clouds and sentence length distributions
- **PyTorch Dataset**: Ready-to-use dataset class for model training

## Project Components

### 1. Introduction to Text Data and Preprocessing
- Download text data from URLs
- Basic text cleaning and normalization
- Sentence splitting using NLTK

### 2. Tokenization and Text Representation
- Word-level tokenization
- Vocabulary creation with unique token IDs
- Token encoding system

### 3. Encoding and Padding
- Encoding tokens to integers
- Padding sequences to fixed lengths
- Truncation for consistent sequence sizes

### 4. Creating a Data Pipeline
- Complete preprocessing pipeline function
- PyTorch `Dataset` class implementation
- Input-target pair generation for next-token prediction

### 5. Visualizing Tokenization and Data
- Sentence length distribution histograms
- Word frequency visualization with word clouds
- Data exploration and analysis

### 6. Complete Text Preprocessing Pipeline
- Modular, reusable preprocessing functions
- Exportable pipeline code
- End-to-end text processing workflow

##  Technologies Used

- **Python 3.x**: Core programming language
- **NLTK (Natural Language Toolkit)**: Sentence tokenization
- **PyTorch**: Dataset class for model training
- **Matplotlib**: Data visualization
- **WordCloud**: Word frequency visualization
- **Regular Expressions (re)**: Text pattern matching and cleaning
- **urllib**: Data downloading from URLs

##  Installation

### Prerequisites

```bash
pip install nltk torch matplotlib wordcloud
```

### NLTK Data Download

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

##  Quick Start

```python
# 1. Install dependencies
!pip install nltk torch matplotlib wordcloud

# 2. Download NLTK data
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# 3. Download dataset
import urllib.request
url = "https://raw.githubusercontent.com/monkeybuzinis/LLM/refs/heads/main/data/ADVENTURES_OF_HUCKLEBERRY_FINN.txt"
urllib.request.urlretrieve(url, "huckleberry_finn.txt")

# 4. Run preprocessing
from text_preprocessing_pipeline import preprocessing_pipeline

with open("huckleberry_finn.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

data = preprocessing_pipeline(raw_text, truncate_len=10)
print(f"Vocabulary size: {len(data['vocab'])}")
print(f"Number of sentences: {len(data['tokenized'])}")
```

##  Usage

### Basic Text Preprocessing

```python
from text_preprocessing_pipeline import preprocessing_pipeline, decode

# Load raw text
with open("huckleberry_finn.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Run preprocessing pipeline
data = preprocessing_pipeline(raw_text, truncate_len=10)

# Access processed data
print("Cleaned text:", data["cleaned_text"][:200])
print("Tokenized sentences:", data["tokenized"][0])
print("Encoded sentences:", data["encoded"][0])
print("Vocabulary size:", len(data["vocab"]))
```

### Creating PyTorch Dataset

```python
from torch.utils.data import DataLoader
from text_preprocessing_pipeline import TokenDataset

# Create dataset
dataset = TokenDataset(data["encoded"])

# Create data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example usage
for x, y in dataloader:
    # x: input tokens (first 9 tokens)
    # y: target tokens (shifted by 1 for next-token prediction)
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
```

### Text Cleaning Function

The `clean_text()` function performs:
- Multiple period normalization (`. . .` → `.`)
- Contraction preservation (don't, can't, won't, etc.)
- Punctuation separation from words
- Whitespace normalization

```python
from text_preprocessing_pipeline import clean_text

# Example 1: Basic cleaning
cleaned = clean_text("Hello, world! Don't worry...")
# Result: "Hello , world ! Don't worry ."

# Example 2: Preserving contractions
text = "I can't believe it's working! We're done."
cleaned = clean_text(text)
# Result: "I can't believe it's working ! We're done ."
```

### Downloading Text Data

```python
import urllib.request

# Download from URL
url = "https://raw.githubusercontent.com/monkeybuzinis/LLM/refs/heads/main/data/ADVENTURES_OF_HUCKLEBERRY_FINN.txt"
file_path = "huckleberry_finn.txt"
urllib.request.urlretrieve(url, file_path)

# Load text
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print(f"Total characters: {len(raw_text)}")
```

### Sentence Tokenization

```python
import nltk
from text_preprocessing_pipeline import clean_text

nltk.download('punkt')

# Clean and tokenize
cleaned_text = clean_text(raw_text)
sentences = nltk.sent_tokenize(cleaned_text)

print(f"Number of sentences: {len(sentences)}")
print(f"First sentence: {sentences[0]}")
```

### Building Vocabulary

```python
# Tokenize all text
all_tokens = cleaned_text.split()
unique_tokens = sorted(set(all_tokens))

# Create vocabulary with special tokens
vocab = {"<UNK>": 0, "<PAD>": 1}
vocab.update({token: idx + 2 for idx, token in enumerate(unique_tokens)})

print(f"Vocabulary size: {len(vocab)}")
print(f"Sample tokens: {list(vocab.items())[:10]}")
```

### Encoding and Decoding

```python
from text_preprocessing_pipeline import preprocessing_pipeline, decode

# Process text
data = preprocessing_pipeline(raw_text, truncate_len=10)

# Encode a sentence
encoded = data["encoded"][0]
print(f"Encoded: {encoded}")

# Decode back to tokens
decoded = decode(encoded, data["inv_vocab"])
print(f"Decoded: {decoded}")
```

##  Dataset Information

- **Primary Dataset**: "The Adventures of Huckleberry Finn" by Mark Twain
- **Total Characters**: ~562,748 characters
- **Total Tokens**: ~133,901 tokens
- **Vocabulary Size**: Varies based on unique tokens
- **Max Sentence Length**: 310 tokens (before truncation)

##  Key Concepts Demonstrated

### 1. Text Cleaning
- Handles contractions intelligently
- Separates punctuation for better tokenization
- Normalizes whitespace and special characters

### 2. Tokenization
- **Sentence-level**: Splits text into sentences using NLTK
- **Word-level**: Splits sentences into individual tokens
- Handles edge cases and special characters

### 3. Vocabulary Building
- Creates unique integer IDs for each token
- Includes special tokens: `<UNK>` (unknown) and `<PAD>` (padding)
- Sorted vocabulary for consistency

### 4. Encoding & Padding
- Converts tokens to integer sequences
- Pads shorter sequences to fixed length
- Truncates longer sequences to maximum length

### 5. Data Pipeline
- Modular design for reusability
- Handles multiple text sources
- Produces model-ready datasets

##  Visualizations

The project includes visualizations for:
- **Sentence Length Distribution**: Histogram showing token count per sentence
- **Word Frequency**: Word cloud visualization of most common words
- **Before/After Truncation**: Comparison of original vs. processed sentence lengths

### Example Visualization Code

```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Sentence length distribution
raw_lengths = [len(sentence.split()) for sentence in raw_sentences]
plt.hist(raw_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title("Original Sentence Token Lengths")
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.show()

# 2. Word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Most Frequent Words")
plt.show()
```

##  Learning Outcomes

After completing this project, you will understand:
- How to preprocess raw text data for NLP tasks
- The importance of proper tokenization and encoding
- How to handle variable-length sequences
- How to create datasets compatible with deep learning frameworks
- Best practices for text cleaning and normalization

##  Project Structure

```
NLP-Text-Processing/
├── NLP-Text-Preprocessing.ipynb
├── huckleberry_finn.txt               
└── README.md                           
```

##  Customization

### Adjusting Sequence Length

```python
# Change truncation length for longer sequences
data = preprocessing_pipeline(raw_text, truncate_len=20)  # 20 tokens instead of 10

# For shorter sequences (faster processing)
data = preprocessing_pipeline(raw_text, truncate_len=5)   # 5 tokens
```

### Adding Custom Special Tokens

```python
# Modify vocabulary building to include custom tokens
all_tokens = cleaned_text.split()
unique_tokens = sorted(set(all_tokens))

# Custom special tokens
vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<START>": 2,
    "<END>": 3,
    "<MASK>": 4,
}

# Add regular tokens
vocab.update({token: idx + 5 for idx, token in enumerate(unique_tokens)})
```

### Processing Multiple Text Files

```python
import glob
from text_preprocessing_pipeline import preprocessing_pipeline

# Process multiple files
text_files = glob.glob("*.txt")
all_data = []

for file_path in text_files:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    data = preprocessing_pipeline(raw_text, truncate_len=10)
    all_data.append(data)

# Combine vocabularies if needed
combined_vocab = {}
for data in all_data:
    combined_vocab.update(data["vocab"])
```

##  Future Enhancements

- [ ] Support for multiple languages
- [ ] Subword tokenization (BPE, SentencePiece)
- [ ] Character-level tokenization option
- [ ] Batch processing for large datasets
- [ ] Integration with Hugging Face tokenizers
- [ ] Support for different text formats (PDF, HTML, etc.)

##  References

- [NLTK Documentation](https://www.nltk.org/)
- [PyTorch Dataset Documentation](https://pytorch.org/docs/stable/data.html)
- [Natural Language Processing with Python](https://www.nltk.org/book/)

## 👤 Author

**Khanh Le**




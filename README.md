# Multimodal Emotion Recognition using RAVDESS Dataset

This project implements two models for emotion classification:
- A CNN-based classifier for audio using mel-spectrograms
- A BERT-based classifier for text using DistilBERT

## Dataset
The model is designed for the [RAVDESS dataset](https://zenodo.org/record/1188976). Emotion labels are extracted from filenames.

### Emotions:
- neutral
- calm
- happy
- sad
- angry
- fearful
- disgust
- surprised

## Code Structure
- `model.py`: Contains dataset preprocessing, CNN and BERT models.
- `requirements.txt`: All required dependencies.
- `README.md`: This file.

## How to Run

1. **Install dependencies:**

```bash
pip install -r requirements.txt

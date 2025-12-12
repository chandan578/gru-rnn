# Next Word Prediction Using GRU

## Project Description

This project develops a deep learning model for predicting the next word in a given sequence using GRU (Gated Recurrent Unit) networks. GRUs are efficient recurrent neural networks ideal for sequence prediction tasks.

## Project Overview

### 1. Data Collection
- Uses Shakespeare's "Hamlet" as the dataset
- Rich, complex text provides an excellent challenge for model training

### 2. Data Preprocessing
- Tokenize text data
- Convert text into sequences
- Pad sequences to uniform input lengths
- Split data into training and testing sets

### 3. Model Building
- Embedding layer for word representation
- GRU layers for sequence learning
- Dense output layer with softmax activation
- Predicts probability distribution for the next word

### 4. Model Training
- Train on prepared sequences
- Implement early stopping to prevent overfitting
- Monitor validation loss for optimal performance

### 5. Model Evaluation
- Evaluate using example sentences
- Test accuracy of next-word predictions

### 6. Deployment
- Streamlit web application for user interaction
- Real-time next-word prediction interface
- Input: sequence of words → Output: predicted next word

## Getting Started

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Technologies Used
- Python
- TensorFlow/Keras
- GRU Networks
- Streamlit
- Pandas, NumPy

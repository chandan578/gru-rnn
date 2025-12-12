import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

## load model
model = load_model('next_word_lstm.h5')

## load tokenizer
with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file=file)


## function to predict the next word

def predict_next_word(model, tokenizer, text, max_length_squence):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_length_squence:
        token_list = token_list[-(max_length_squence-1):] 
    token_list = pad_sequences([token_list], maxlen=max_length_squence-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

## streamlit app
st.title("Next word prediction with LSTM and Earlystopping")
input_text = st.text_input('Enter the sequence of words', 'To be or not to be')
if st.button('Predict Next word'):
    max_seq_len = model.input_shape[1]+1
    next_word = predict_next_word(model=model, tokenizer=tokenizer, text=input_text, max_length_squence=max_seq_len)
    st.write(f"Next word : {next_word}")
else :
    st.write('Enter something else...')
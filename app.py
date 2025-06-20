from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

app = Flask(__name__)

# Load model and tokenizer
model = load_model('project.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

sequence_length = 40  # same as your training

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

def predict_next_words(seed_text, num_words=5, temperature=0.5):
    result = seed_text
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding='pre')
        preds = model.predict(token_list, verbose=0)[0]
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        next_index = np.random.choice(len(preds), p=preds)
        next_word = tokenizer.index_word.get(next_index, '')
        if not next_word or next_word == '<OOV>':
            break
        result += ' ' + next_word
    return result

@app.route('/', methods=['GET', 'POST'])
def home():
    predictions = []
    if request.method == 'POST':
        seed_text = request.form['seed_text']
        seed_text = clean_text(seed_text)
        predictions = [predict_next_words(seed_text, num_words=5) for _ in range(3)]
    return render_template('index.html', predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True)

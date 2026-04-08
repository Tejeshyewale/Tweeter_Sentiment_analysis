from fastapi import FastAPI
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocess import preprocess

app = FastAPI()

model = pickle.load(open('../models/model.pkl', 'rb'))
vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))

@app.get("/")
def home():
    return {"message": "Sentiment API Running"}

@app.post("/predict")
def predict(text: str):
    processed = preprocess(text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)

    return {
        "text": text,
        "sentiment": "Positive" if prediction[0] == 1 else "Negative"
    }
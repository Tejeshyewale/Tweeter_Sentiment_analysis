import pickle
from preprocess import preprocess

# Load model
model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

def predict_sentiment(text):
    processed = preprocess(text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)

    return "Positive 😊" if prediction[0] == 1 else "Negative 😡"
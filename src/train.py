import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import preprocess

# ===============================
# 📁 Set base path (IMPORTANT)
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'training.1600000.processed.noemoticon.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create models folder
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# 📊 Load dataset
# ===============================
df = pd.read_csv(DATA_PATH, encoding='latin-1', header=None)

df.columns = ['target','id','date','flag','user','text']
df = df[['target','text']]

# Convert labels
df['target'] = df['target'].replace({4:1})

print("Preprocessing text...")

# ===============================
# 🧹 Preprocess
# ===============================
df['clean_text'] = df['text'].apply(preprocess)

# ===============================
# 🔢 Feature Engineering
# ===============================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['target']

# ===============================
# ✂️ Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training model...")

# ===============================
# 🤖 Model Training
# ===============================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ===============================
# 💾 Save Model (CORRECT PATH)
# ===============================
pickle.dump(model, open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb'))
pickle.dump(vectorizer, open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'wb'))

print("✅ Model trained and saved successfully!")
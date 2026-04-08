# 🐦 Twitter Sentiment Analyzer

🔗 **Live App:** [https://tweetersentimentanalysis-6hytugjqqjfnvufm2uognw.streamlit.app/](https://tweetersentimentanalysis-6hytugjqqjfnvufm2uognw.streamlit.app/)

---

## 🚀 Overview

Twitter Sentiment Analyzer is an end-to-end NLP-based web application that analyzes the sentiment of tweets in real-time.

It classifies user input text into:

* 😊 Positive
* 😡 Negative

with confidence scores and interactive visualizations.

---

## ✨ Features

* 🔍 Real-time sentiment prediction
* 📊 Interactive dashboard with charts
* 📈 Sentiment distribution (donut chart)
* 📉 Confidence tracking over time
* 🧠 NLP preprocessing (cleaning, stemming, stopwords removal)
* 💾 History tracking of analyzed tweets
* 🎨 Modern UI using Streamlit + custom CSS

---

## 🧠 Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Machine Learning:**

  * TF-IDF Vectorization
  * Logistic Regression
* **NLP:** NLTK
* **Visualization:** Plotly

---

## 📂 Project Structure

```
tweeter_sentiment_analysis/
│
├── app/
│   └── app.py
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Tejeshyewale/Tweeter_Sentiment_analysis.git
cd Tweeter_Sentiment_analysis
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Train model (if needed)

```bash
python src/train.py
```

---

### 4️⃣ Run the app

```bash
streamlit run app/app.py
```

---

## 📸 Demo

> Add screenshots of your UI here (recommended)

---

## 🎯 Results

* Real-time sentiment classification
* High accuracy using TF-IDF + Logistic Regression
* Smooth interactive dashboard

---

## 💡 Future Improvements

* 🔄 Add Neutral sentiment class
* 🐦 Integrate Twitter API for live tweets
* ☁️ Deploy on AWS / Docker
* 🤖 Upgrade to BERT / Transformer models

---

## 👨‍💻 Author

**Tejesh Yewale**

* Data Science Enthusiast 🚀
* AWS & Machine Learning Learner

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

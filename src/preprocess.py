import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Remove special characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Lowercase
    text = text.lower()
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords + stemming
    words = [ps.stem(word) for word in words if word not in stop_words]
    
    # Join back
    return ' '.join(words)
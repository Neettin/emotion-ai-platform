import string
import nltk
from nltk.corpus import stopwords

# Streamlit Cloud cache for NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Complete preprocessing pipeline used during training"""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    # 4. Remove stopwords
    words = text.split()
    cleaned = [word for word in words if word not in stop_words]
    return ' '.join(cleaned)
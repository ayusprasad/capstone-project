import joblib
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Create a new vectorizer with the same parameters
vectorizer = CountVectorizer(max_features=5000)

# Save the vectorizer
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
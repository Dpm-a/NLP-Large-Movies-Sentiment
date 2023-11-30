import re
import numpy
import joblib
import argparse
import numpy as np
import pandas as pd
import contractions

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import AdamW


MAX_LENGTH = 400
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))
BEST_MODEL = joblib.load('../model/best_svc_model.pkl')
test_df = pd.read_csv("../data/test.csv")

# Model parameters
embedding_dim = 128
lstm_units = 128


# cleaning different patterns
def clean_review(text, bert=False):
    """
    It cleans the tokens from unrelevant characters
    """
    text = text.strip().lower()
    text = contractions.fix(text)
    text = re.sub(r'<.*?>', '', text)  # removing HTMLS
    # removing urls and emails
    text = re.sub(r'http\S+|www\S+|email', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)

    if not bert:
        words = word_tokenize(text)  # Tokenize text
        # Lemmatize and remove stop words
        cleaned_text = [LEMMATIZER.lemmatize(
            word) for word in words if word not in STOP_WORDS]
        text = ' '.join(cleaned_text)  # Join words back to string

    return text


def main(single_sentence):

    with open('tokenizer.json') as f:
        data = f.read()
        tokenizer = tokenizer_from_json(data)

    if single_sentence:
        test_sequences = tokenizer.texts_to_sequences(single_sentence)
        test_padded = pad_sequences(
            test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
        sentiment = np.round(BEST_MODEL.predict(test_padded)).flatten()
        print(f"Sentiment: {sentiment}")
        return

    test_df["rating"] = test_df["rating"].apply(lambda x: 1 if x > 5 else 0)
    test_df["lemmas"] = test_df["review"].apply(lambda r: clean_review(r))
    y_test = test_df["rating"].values

    # setting up tokenizer
    test_sequences = tokenizer.texts_to_sequences(test_df["lemmas"])
    test_padded = pad_sequences(
        test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

    y_pred = np.round(BEST_MODEL.predict(test_padded)).flatten()
    test_acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy on Tets set: {test_acc}\n")
    print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on Test set")
    parser.add_argument("-s", "--sentence",
                        help="Inference on a single sentence", default=False)

    args = parser.parse_args
    main(args.sentence)

import re
import numpy as np
import joblib
import argparse
import numpy as np
import pandas as pd
import contractions
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.preprocessing.text import tokenizer_from_json

from keras.models import load_model
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import torch
from dataset import IMDbDataset
from transformers import BertTokenizer, BertForSequenceClassification


MAX_LENGTH = 200
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))
# BEST_MODEL = joblib.load('../models/best_svc_model.pkl')
BEST_MODEL = load_model('../models/lstm_200.h5')
TEST_DF = pd.read_csv("../data/test.csv")
BERT = BertForSequenceClassification.from_pretrained(
    '../models/transformers_3')


# Function to tokenize the dataset
def tokenize_data(dataframe, tokenizer):
    if isinstance(dataframe, pd.DataFrame):
        return tokenizer(list(dataframe["clean_review"]), padding=True, truncation=True, max_length=512)
    return tokenizer(list(dataframe.values), padding=True, truncation=True, max_length=512)

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


# Function to prepare a single sentence for the model
def prepare_single_sentence(sentence, tokenizer, max_length=512):
    inputs = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
        return_tensors="pt"  # Return PyTorch tensors
    )
    return inputs['input_ids'], inputs['attention_mask']


def predict(model, data, device, is_single_sentence=False):
    model.to(device)
    model.eval()
    predictions = []

    if is_single_sentence:
        input_ids, attention_mask = data
        input_ids, attention_mask = input_ids.to(
            device), attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_class = torch.argmax(outputs.logits, dim=1).cpu().item()
            return predicted_class
    else:
        data_loader = DataLoader(data, batch_size=32, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids, attention_mask = batch['input_ids'].to(
                    device), batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                predictions.extend(torch.argmax(
                    outputs.logits, dim=1).cpu().tolist())

        return predictions


def main(single_sentence, x_test_col, y_test_col, transformers):

    if not any(el in TEST_DF.columns for el in [x_test_col, y_test_col]):
        raise ValueError("""
                         Please set up correct X and Y column names
                         Example: 'inference.py -x text -y sentiment
                         Base: X = "review", Y = "rating"
                         """)

    if single_sentence or transformers:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        BERT.to(device)

        # For a single sentence
        if single_sentence:
            single_input_ids, single_attention_mask = prepare_single_sentence(
                single_sentence, tokenizer)
            predicted_class = predict(
                BERT, (single_input_ids, single_attention_mask), device, is_single_sentence=True)
            print("Predicted class for single sentence:", predicted_class)
            return

        # For a dataset
        print("- Cleaning Reviwes -")
        TEST_DF["clean_review"] = TEST_DF[x_test_col].apply(
            lambda r: clean_review(r, bert=True))
        TEST_DF[y_test_col] = TEST_DF[y_test_col].apply(
            lambda x: 1 if x > 5 else 0)
        y_test = TEST_DF[y_test_col].values

        print("- Encoding Dataset -")
        test_encodings = tokenize_data(TEST_DF["clean_review"], tokenizer)
        test_dataset = IMDbDataset(test_encodings, TEST_DF['rating'].tolist())
        print("- Inferencing on Input Dataset -")
        predictions = predict(BERT, test_dataset, device)

        report = classification_report(y_test, predictions, target_names=[
                                       'Class 0', 'Class 1'], digits=3)
        print(report)
        return

    ######### USING LSTM ##############
    with open('../models/tokenizer.json') as f:
        data = f.read()
        tokenizer = tokenizer_from_json(data)

    TEST_DF["rating"] = TEST_DF[y_test_col].apply(lambda x: 1 if x > 5 else 0)
    TEST_DF["lemmas"] = TEST_DF[x_test_col].apply(lambda r: clean_review(r))
    y_test = TEST_DF[y_test_col].values

    # setting up tokenizer
    test_sequences = tokenizer.texts_to_sequences(TEST_DF["lemmas"])
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
    parser.add_argument("-x", "--x_test",
                        help="explicit set the column to accept as input data", default="review")
    parser.add_argument("-y", "--y_test",
                        help="explicit set the column to accept as target variable", default="rating")
    parser.add_argument("-t", "--transformers",
                        help="Inference with transformers instead", action="store_true")

    args = parser.parse_args()
    main(args.sentence,
         str(args.x_test),
         str(args.y_test),
         args.transformers)

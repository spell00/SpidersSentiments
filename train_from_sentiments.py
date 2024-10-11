import pandas as pd
import numpy as np
import os
import requests
from transformers import pipeline
from bs4 import BeautifulSoup
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
import spacy
from preprocess import SymSpell, preprocessing
from utils import now, format_delta, split_into_sentences
import pickle
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
from sklearn.metrics import accuracy_score
from selenium.common.exceptions import NoSuchElementException, WebDriverException

from bs4 import BeautifulSoup as bs
from selenium import webdriver
import requests
from selenium.webdriver.firefox.options import Options as FirefoxOptions

import evaluate

from transformers import Trainer, TrainingArguments, LongformerForSequenceClassification, EarlyStoppingCallback, pipeline

from datasets import Dataset

accuracy_metric = evaluate.load("accuracy")
mcc_metric = evaluate.load("matthews_correlation")

# Use options to have your selenium headless
options = FirefoxOptions()
options.add_argument("--headless")
driver = webdriver.Firefox(options=options)


# Charger le CSV avec pandas
def charger_csv(chemin_csv):
    return pd.read_csv(chemin_csv, sep='\t')

# Obtenir le texte de l'URL avec BeautifulSoup
def obtenir_texte(url):
    try:
        reponse = requests.get(url)
        reponse.raise_for_status()
        soup = BeautifulSoup(reponse.text, 'html.parser')
        texte = soup.get_text()
        return texte
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la récupération de l'URL {url}: {e}")
        return ""

# requests_session = requests.Session()
def obtenir_texte_better_but_slow(url):
    try:
        driver.get(url)
        page = driver.page_source
        soup = bs(page, 'lxml')
        texte = soup.get_text()
        return texte
    except (NoSuchElementException, WebDriverException) as e:
        print(f"Erreur lors de la récupération de l'URL {url}: {e}")
        return ""

# Analyser les sentiments avec le modèle Hugging Face
def analyser_sentiments(texte, sentiment_analyzer):
    # resultat_sentiment.save_model("data/berteet_sentiment_analysis.model")
    resultat_sentiment = sentiment_analyzer(texte)
    return resultat_sentiment[0]

def preprocess(texte, stopwords, ss, words_dict):
    texte = ' '.join(nltk.word_tokenize(texte)).lower()
    if stopwords is not None:
        texte = preprocessing(texte, stopwords, ss, words_dict=words_dict)
    return texte

from transformers import LongformerTokenizer
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
# Define a function to tokenize the articles
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=4096)

class HuggingFaceClassifier:
    def __init__(self, model=None):
        if model is not None:
            self.crop = -1
            self.classifier = pipeline('sentiment-analysis', model=model)
        else:
            self.crop = 399
            self.classifier = pipeline('sentiment-analysis')
        # self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        # self.config = AutoConfig.from_pretrained(MODEL)
        # # PT
        # self.classifier = AutoModelForSequenceClassification.from_pretrained(MODEL)


    def predict(self, texte):
        dic = self.classifier(texte[:self.crop])
        return dic[0]['label'], dic[0]['score']

def get_trainer(classif, tokenized_datasets):
    #  Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",               # output directory
        evaluation_strategy="epoch",          # evaluate at the end of each epoch
        learning_rate=2e-5,                   # learning rate
        per_device_train_batch_size=2,        # batch size for training
        per_device_eval_batch_size=2,         # batch size for evaluation
        num_train_epochs=3,                   # number of training epochs
        weight_decay=0.01,                    # strength of weight decay
        logging_dir="./logs",                 # directory for storing logs
        logging_steps=10,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=classif,                          # the instantiated 🤗 Transformers model
        args=training_args,                   # training arguments, defined above
        train_dataset=tokenized_datasets["train"],  # training dataset
        eval_dataset=tokenized_datasets["test"],    # evaluation dataset
    )

    # Train the model
    trainer.train()

    if classif == 'huggingface_binary':
        return HuggingFaceClassifier()
    elif classif == 'huggingface_bertweet':
        return HuggingFaceClassifier(model="finiteautomata/bertweet-base-sentiment-analysis")
    elif classif == 'huggingface_roberta':
        return HuggingFaceClassifier(model=f"cardiffnlp/twitter-roberta-base-sentiment-latest")

def get_symspell(eng_words):
    if 'symspell.pkl' in os.listdir('.'):
        # load pickle
        filehandler = open('symspell.pkl', 'rb')
        ss = pickle.load(filehandler)
    else:
        print('Creating dictionary for symspell')
        begin = now()
        ss = SymSpell(max_edit_distance=2)
        _ = ss.create_dictionary_from_arr(eng_words, token_pattern=r'.+')
        filehandler = open('symspell.pkl', 'wb')
        pickle.dump(ss, filehandler)
        print('Finished dictionary for symspell in', format_delta(begin, now()))
    return ss

def get_eng_words(file='data/english_words_479k.txt'):
    with open(file) as f:
        words = f.readlines()
    eng_words = [word.strip() for word in words]
    return eng_words


# Define the function that computes metrics (accuracy and MCC)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get the predicted class
    
    # Compute accuracy
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels.argmax(axis=1))
    
    # Compute MCC
    mcc = mcc_metric.compute(predictions=predictions, references=labels.argmax(axis=1))

    return {
        'accuracy': accuracy['accuracy'],
        'mcc': mcc['matthews_correlation']
    }

# Fonction principale
def main(args):

    dataset = Dataset.load_from_disk('data/dataset_Sensationalism.hf')
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Split the dataset into training and testing sets
    train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    # Verify the number of samples in the original dataset
    total_samples = len(dataset)
    print(f"Total samples in the dataset: {total_samples}")

    # After splitting into train and test sets
    train_size = len(train_dataset)
    test_size = len(test_dataset)

    print(f"Number of samples in the training set: {train_size}")
    print(f"Number of samples in the testing set: {test_size}")
    #  Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",               # output directory
        evaluation_strategy="epoch",          # evaluate at the end of each epoch
        save_strategy="epoch",                # save model at the end of each epoch to match eval strategy
        learning_rate=2e-5,                   # learning rate
        per_device_train_batch_size=8,        # batch size for training
        per_device_eval_batch_size=8,         # batch size for evaluation
        num_train_epochs=1000,                # number of training epochs
        weight_decay=0.01,                    # strength of weight decay
        logging_dir="./logs",                 # directory for storing logs
        logging_steps=10,
        load_best_model_at_end=True,          # Load the best model at the end of training
        save_total_limit=2,                   # Limit saved checkpoints to 2
    )
    # model = HuggingFaceClassifier(model=f"cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = LongformerForSequenceClassification.from_pretrained(
        "allenai/longformer-base-4096", num_labels=2
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,                          # the instantiated 🤗 Transformers model
        args=training_args,                   # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,    # evaluation dataset
        compute_metrics=compute_metrics,  # Add compute_metrics to Trainer
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stops after 3 non-improving evals
    )

    # Train the model
    trainer.train()
    trainer.save_model("longformer-base-4096.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--classif", type=str, default='huggingface_roberta')
    parser.add_argument("--use_preprocess", type=int, default=0)
    parser.add_argument("--column", type=str, default='Sensationalism')
    parser.add_argument("--test", type=bool, default=False)
    args = parser.parse_args()

    main(args)


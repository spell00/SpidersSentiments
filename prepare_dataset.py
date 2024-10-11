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

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig

from transformers import Trainer, TrainingArguments

from datasets import Dataset


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

def label_to_onehot(label):
    # Create a one-hot vector of length 2
    onehot_vector = np.zeros(2)
    
    # Set the corresponding index to 1 based on the label
    onehot_vector[label] = 1
    
    return onehot_vector

# Fonction principale
def main(args):

    n_found = 0
    data_frame = charger_csv(args.chemin_csv)
    new_df = []
    articles = []

    # Ajouter une nouvelle colonne pour les résultats d'analyse des sentiments
    # data_frame['POS'] = ''
    # data_frame['NEG'] = ''
    if args.use_preprocess:
        eng_words = get_eng_words()
        ss = get_symspell(eng_words)
        words_dict = {k: 0 for k in eng_words}
        spacy_nlp = spacy.load('en_core_web_lg')
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        stop_words = set(list(set(stopwords.words('english'))) + list(spacy_stopwords))
    else:
        stop_words = None
        words_dict = None
        ss = None

    for index, ligne in tqdm(data_frame.iterrows(), total=len(data_frame), desc="Preparation du dataset"):
        if np.isnan(ligne[args.column]):
            continue
        url = ligne['URL']
        lang = ligne['Language']
        if lang != 'English':
            continue
        else:
            n_found += 1
            print('found.', lang, n_found)

        texte = obtenir_texte_better_but_slow(url)
        texte = preprocess(texte, stop_words, ss, words_dict)
        # Make label one hot encoded
        articles += [
            {
                'text': texte,
                'label': label_to_onehot(int(ligne[args.column]))
            }
        ]
        if args.test and len(articles) > 10:
            break

    # Convert the list of articles into a dataset
    dataset = Dataset.from_dict({"text": [article["text"] for article in articles], 
                                "label": [article["label"] for article in articles]})
    dataset.save_to_disk(f"data/dataset_{args.column}.hf")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemin_csv", type=str, default="data/Data_spider_news_global.csv")
    parser.add_argument("--use_preprocess", type=int, default=0)
    parser.add_argument("--column", type=str, default='Sensationalism')
    parser.add_argument("--test", type=int, default=0)
    args = parser.parse_args()

    main(args)


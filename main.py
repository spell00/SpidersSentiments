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

# Use options to have your selenium headless
options = FirefoxOptions()
options.add_argument("--headless")
driver = webdriver.Firefox(options=options)

# url = "https://www.businessinsider.sg/kim-kardashian-found-tarantulas-in-her-house-2019-8"
# try:
#     driver.get(url)
#     page = driver.page_source
#     soup = bs(page, 'lxml')
#     texte = soup.get_text()
# except (NoSuchElementException, WebDriverException) as e:
#     print(f"Erreur lors de la récupération de l'URL {url}: {e}")
# 
# page = driver.page_source
# soup = bs(page, 'html.parser')
# texte = soup.get_text()


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

# def obtenir_texte2(url):
#     try:
#         reponse = requests_session.get(url)
#         reponse.raise_for_status()
#         soup = bs(reponse.text, 'lxml')
#         texte = soup.get_text()
#         return texte
#     except requests.exceptions.RequestException as e:
#         print(f"Erreur lors de la récupération de l'URL {url}: {e}")
#         return ""

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

class FlairClassifier:
    def __init__(self):
        self.classifier = TextClassifier.load('en-sentiment')
    def predict(self, texte):
        sentence = Sentence(texte)
        self.classifier.predict(sentence)
        return sentence.labels[0].value, sentence.labels[0].score

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

class VaderClassifier:
    def __init__(self):
        self.classifier = SentimentIntensityAnalyzer()
    def predict(self, texte):
        scores = self.classifier.polarity_scores(texte)
        if scores['compound'] >= 0.05:
            return 'POSITIVE', scores['compound']
        elif scores['compound'] <= -0.05:
            return 'NEGATIVE', scores['compound']
        else:
            return 'NEUTRAL', scores['compound']
        

def get_classifier(classif):
    if classif == 'flair':
        return FlairClassifier()
    elif classif == 'huggingface_binary':
        return HuggingFaceClassifier()
    elif classif == 'huggingface_bertweet':
        return HuggingFaceClassifier(model="finiteautomata/bertweet-base-sentiment-analysis")
    elif classif == 'huggingface_roberta':
        return HuggingFaceClassifier(model=f"cardiffnlp/twitter-roberta-base-sentiment-latest")
    elif classif == 'vader':
        return VaderClassifier()

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

# Fonction principale
def main(chemin_csv, classif, use_preprocess):

    n_found = 0
    classifier = get_classifier(classif)    
    data_frame = charger_csv(chemin_csv)
    new_df = []

    # Ajouter une nouvelle colonne pour les résultats d'analyse des sentiments
    # data_frame['POS'] = ''
    # data_frame['NEG'] = ''
    if use_preprocess:
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
        
    for index, ligne in tqdm(data_frame.iterrows(), total=len(data_frame), desc="Analyse des sentiments"):
        url = ligne['URL']
        lang = ligne['Language']
        if lang != 'English':
            continue
        else:
            n_found += 1
            print('found.', lang, n_found)
        
        texte = obtenir_texte_better_but_slow(url)
        texte = preprocess(texte, stop_words, ss, words_dict)
        positives = []
        negatives = []
        neutrals = []
        sentences = split_into_sentences(texte)

        for i, sent in enumerate(sentences):
            try:
                value, score = classifier.predict(sent)
            except:
                print("Problem with sentence. Probabiliy contains words too long > 128")
                continue
            
            if value in ['POSITIVE', 'positive']:
                positives.append(sent)
            elif value in ['NEGATIVE', 'negative']:
                negatives.append(sent)

        if len(sentences) == 0:
            continue
        else:
            pos_rate = len(positives) / len(sentences)
            neg_rate = len(negatives) / len(sentences)
            rates = [pos_rate, neg_rate]
        
        new_df.append(np.concatenate([ligne.values, rates]))

    # Enregistrer le résultat dans un nouveau CSV
    new_df = pd.DataFrame(np.stack(new_df))
    new_df.columns = data_frame.columns.tolist() + ['POS', 'NEG']
    os.makedirs(f'resultats/preprocess{use_preprocess}/', exist_ok=True)
    new_df.to_csv(
        f'resultats/preprocess{use_preprocess}/resultats_sentiments_{classif}.csv',
          index=False)
    print('DONE. Found:', n_found, 'English articles.')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemin_csv", type=str, default="data/Data_spider_news_global.csv")
    parser.add_argument("--classif", type=str, default='huggingface_roberta')
    parser.add_argument("--use_preprocess", type=int, default=1)
    args = parser.parse_args()

    main(
        args.chemin_csv,
        args.classif,
        args.use_preprocess
    )

from spider_guardian.langsmith.simple import push_reply_to_dataset
# Utility to log a reply to LangSmith dataset
def log_reply_to_langsmith_dataset(
    tweet_text,
    author,
    url,
    generated_reply,
    likes=0,
    replies=0,
    impressions=0,
    metadata=None
):
    """Log a reply (bot or human) to the LangSmith dataset."""
    push_reply_to_dataset(
        tweet_text=tweet_text,
        author=author,
        url=url,
        generated_reply=generated_reply,
        likes=likes,
        replies=replies,
        impressions=impressions,
        metadata=metadata or {}
    )
# Example usage (call this after posting or collecting a reply)
# log_reply_to_langsmith_dataset(
#     tweet_text="Why do spiders bite?",
#     author="user123",
#     url="https://twitter.com/user123/status/789",
#     generated_reply="Spiders bite mostly in self-defense, and most bites are harmless.",
#     likes=3,
#     replies=0,
#     impressions=50,
#     metadata={"source": "bot"}
# )
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
from datetime import datetime

from spider_guardian.storage import SQLDataStore, SentimentResult

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


# Charger le dataset depuis SQL ou CSV
def charger_dataset(chemin_csv: str, sql_db: str, delimiter: str = '\t') -> pd.DataFrame:
    if sql_db:
        store = SQLDataStore(sql_db)
        try:
            df = store.dataset_dataframe()
            if not df.empty:
                return df
        finally:
            store.close()
    return pd.read_csv(chemin_csv, sep=delimiter)

# Obtenir le texte de l'URL avec BeautifulSoup
def obtenir_texte(url):
    try:
        reponse = requests.get(url)
        reponse.raise_for_status()
        soup = BeautifulSoup(reponse.text, 'html.parser')
        texte = soup.get_text()
        # Log the collected text as a human reply to LangSmith
        push_reply_to_dataset(
            tweet_text=texte[:200],  # Truncate for dataset
            author="human_collector",
            url=url,
            generated_reply="",  # No generated reply, just collected
            metadata={"source": "human", "function": "obtenir_texte"}
        )
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
        # Log the collected text as a human reply to LangSmith
        push_reply_to_dataset(
            tweet_text=texte[:200],
            author="human_collector",
            url=url,
            generated_reply="",
            metadata={"source": "human", "function": "obtenir_texte_better_but_slow"}
        )
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
    resultat_sentiment = sentiment_analyzer(texte)
    # Log sentiment analysis as a bot reply (example)
    push_reply_to_dataset(
        tweet_text=texte, # [:200]
        author="bot",
        url="",
        generated_reply=str(resultat_sentiment[0]),
        metadata={"source": "bot", "function": "analyser_sentiments"}
    )
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
        score = sentence.labels[0].score
        value = sentence.labels[0].value
        if score >= 0.8:
            return value, score
        else:
            return 'NEUTRAL', score

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
def main(
    chemin_csv,
    classif,
    use_preprocess,
    sql_db,
    delimiter='\t',
    legacy_csv=None,
):

    n_found = 0
    classifier = get_classifier(classif)
    data_frame = charger_dataset(chemin_csv, sql_db, delimiter)
    new_df = []
    sentiment_results = []
    store = SQLDataStore(sql_db) if sql_db else None

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
            else:
                neutrals.append(sent)

        if len(sentences) == 0:
            continue
        else:
            pos_rate = len(positives) / len(sentences)
            neg_rate = len(negatives) / len(sentences)
            neu_rate = len(neutrals) / len(sentences)
            rates = [pos_rate, neg_rate, neu_rate]

        new_df.append(np.concatenate([ligne.values, rates]))

        if store is not None:
            payload = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in ligne.to_dict().items()}
            payload.update({'POS': pos_rate, 'NEG': neg_rate, 'NEU': neu_rate})
            dataset_id = str(payload.get('ID', index))
            sentiment_results.append(
                SentimentResult(
                    dataset_id=dataset_id,
                    classifier=classif,
                    preprocess=int(use_preprocess),
                    pos=float(pos_rate),
                    neg=float(neg_rate),
                    neu=float(neu_rate),
                    created_at=datetime.utcnow(),
                    payload=payload,
                )
            )

    # Enregistrer le résultat dans un nouveau CSV
    if new_df:
        new_df = pd.DataFrame(np.stack(new_df))
        new_df.columns = data_frame.columns.tolist() + ['POS', 'NEG', 'NEU']
        target_csv = legacy_csv or f'resultats/preprocess{use_preprocess}/resultats_sentiments_{classif}.csv'
        os.makedirs(os.path.dirname(target_csv) or '.', exist_ok=True)
        new_df.to_csv(target_csv, index=False)
    else:
        new_df = pd.DataFrame(columns=data_frame.columns.tolist() + ['POS', 'NEG', 'NEU'])

    if store is not None:
        if sentiment_results:
            store.clear_sentiment_results(classifier=classif, preprocess=int(use_preprocess))
            store.save_sentiment_results(sentiment_results)
        store.close()
    print('DONE. Found:', n_found, 'English articles.')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemin_csv", type=str, default="data/Data_spider_news_global.csv")
    parser.add_argument("--classif", type=str, default='huggingface_roberta')
    parser.add_argument("--use_preprocess", type=int, default=0)
    parser.add_argument("--sql_db", type=str, default="data/spider_guardian.sqlite")
    parser.add_argument("--delimiter", type=str, default='\t')
    parser.add_argument("--legacy_csv", type=str, default=None)
    args = parser.parse_args()

    main(
        args.chemin_csv,
        args.classif,
        args.use_preprocess,
        args.sql_db,
        args.delimiter,
        args.legacy_csv,
    )

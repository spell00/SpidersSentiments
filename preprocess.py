# Code adapté à partir de : https://www.kaggle.com/saxinou/nlp-01-preprocessing-data

# symspell: https://www.kaggle.com/yk1598/symspell-spell-corrector, lui-meme basé sur
# https://www.kaggle.com/rumbok/ridge-lb-0-41944
"""
A RAJOUTER:

    1- Remplacer numeros qui devraient etre des lettres et autres 'leetspeak' https://simple.wikipedia.org/wiki/Leet
        0 -> o (n00b -> noob)
        1 -> l
        2 -> Z or e
        3 -> E (n33d -> need)
        4 ->
        5 -> S
        6 -> b
        7 -> T or j
        8 -> B or X
        9 -> g or j

"""
import itertools
import string
import nltk
from collections import Counter
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import unidecode
from utils import now, format_delta
import pandas as pd
from tqdm import tqdm
import emoji

CONTRACTION_MAP = {"ain't": "is not", "aren't": "are not", "can't": "cannot",
                   "can't've": "cannot have", "'cause": "because", "could've": "could have",
                   "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
                   "he'll've": "he he will have", "he's": "he is", "how'd": "how did",
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                   "I'll've": "I will have", "I'm": "I am", "I've": "I have",
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                   "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                   "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                   "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                   "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                   "she's": "she is", "should've": "should have", "shouldn't": "should not",
                   "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                   "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                   "they'll've": "they will have", "they're": "they are", "they've": "they have",
                   "to've": "to have", "wasn't": "was not", "we'd": "we would",
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                   "we're": "we are", "we've": "we have", "weren't": "were not",
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                   "what's": "what is", "what've": "what have", "when's": "when is",
                   "when've": "when have", "where'd": "where did", "where's": "where is",
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                   "who's": "who is", "who've": "who have", "why's": "why is",
                   "why've": "why have", "will've": "will have", "won't": "will not",
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                   "you'll've": "you will have", "you're": "you are", "you've": "you have"}


def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.
    This method has not been modified from the original.
    Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/
    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.
    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.
    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.
    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2
    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


def tokenize_word_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def convert_letters(tokens, style="lower"):
    if (style == "lower"):
        tokens = [token.lower() for token in tokens]
    else:
        tokens = [token.upper() for token in tokens]
    return (tokens)


def remove_blanc(tokens):
    tokens = [token.strip() for token in tokens]
    return (tokens)


def replace_numeric(sent,
                    numeric_pattern=re.compile('[0-9]+'),
                    digit_pattern=re.compile('[0-9]'), repl='#',
                    by_single_digit=True):
    return re.sub(numeric_pattern, repl, sent) if by_single_digit else re.sub(digit_pattern, repl, sent)


def remove_before_token(sentence, keep_apostrophe=False):
    sentence = sentence.strip()
    if keep_apostrophe:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]'
        filtered_sentence = re.sub(PATTERN, r' ', sentence)
    else:
        PATTERN = r'[^a-zA-Z0-9]'
        filtered_sentence = re.sub(PATTERN, r' ', sentence)
    return (filtered_sentence)


def remove_after_token(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def expand_contractions(sentence, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(
            match) else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence


def remove_accent_before_tokens(sentences):
    res = unidecode.unidecode(sentences)
    return (res)


def remove_accent_after_tokens(tokens):
    tokens = [unidecode.unidecode(token) for token in tokens]
    return (tokens)


def removeStopwords_after_tokens(wordlist, stopwords):
    return [w for w in wordlist if w not in stopwords]


def removeStopwords_before_tokens(text, stopwords):
    tokens = nltk.word_tokenize(text)
    return [w for w in tokens if w not in stopwords]


def words(text): return re.findall(r'\w+', text.lower())


inputs = open('data/big.txt', encoding="utf8").read()
WORDS = Counter(words(inputs))


def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N


def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)


def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def correct_word_in_sentence(text):
    tokens = nltk.word_tokenize(text)
    r = [correction(token) for token in tokens]
    return (r)


def replace_numeric(sent,
                    numeric_pattern=re.compile('[0-9]+'),
                    digit_pattern=re.compile('[0-9]'), repl='#',
                    by_single_digit=True):
    return re.sub(numeric_pattern, repl, sent) if by_single_digit else re.sub(digit_pattern, repl, sent)


def replace_stopwords(text, stop_words):
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


# This function removes every punctuation character. I don't
def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))


# With word_tokenize, every punctuation character is considered as a word.
# This function removes every duplicates word (and so ponctuation signs)
def normalize_punctuation_and_repeated_words(list_of_tokenized_words):
    return [x for x, y in itertools.groupby(list_of_tokenized_words)]


def remove_consecutive_duplicates_character_in_a_word(word):
    characters = list(word)
    return [char[0] for char in itertools.groupby(characters)]


# Very simple, but efficient.
def normalize_laughting(list_of_tokenized_words):
    for indexWord in range(0, len(list_of_tokenized_words)):
        word = list_of_tokenized_words[indexWord]
        if ("ahah" in word or "haha" in word or "lol" in word or "xd" in word):
            list_of_tokenized_words[indexWord] = "RIRE"

    return list_of_tokenized_words


def normalize_time(list_of_tokenized_words):
    from dateutil import parser

    for indexWord in range(0, len(list_of_tokenized_words)):
        word = list_of_tokenized_words[indexWord]
        # I know this code isnt beautiful, but it is the more efficient way to find, in a string, if a word represented a date, or a time.
        try:
            parser.parse(word).strftime("%H:%M")

            list_of_tokenized_words[indexWord] = "TIME"
        except ValueError:
            pass

    return list_of_tokenized_words


# To make a perfect correction would be much too long. Here, one is satisfied with an imperfect correction hoping that the corrector corrects the error.
def remove_multiple_characters(sentence):
    return re.sub(r'(.)\1+', r'\1\1', sentence)


# Returns thumbs_up for thumbs up, for example.
def normalize_smileys_in_sentence(sentence):
    demojized_sentence = emoji.demojize(sentence).replace(":", " ")
    matches = re.findall(r'\:(.+?)\:', demojized_sentence)

    for m in matches:
        demojized_sentence = demojized_sentence.replace(":", "")

    return demojized_sentence


class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose
        # 0: top suggestion
        # 1: all suggestions of smallest edit distance
        # 2: all suggestions <= max_edit_distance (slower, no early termination)

        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        """given a word, derive strings with up to max_edit_distance characters
           deleted"""

        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word
            # (deleting character from a real word)
            # but counter of frequency of word in corpus is not incremented
            # in those cases)
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added

    def create_dictionary_from_arr(self, arr, token_pattern=r'[a-z]+'):
        total_word_count = 0
        unique_word_count = 0

        n_line = 0
        print("Creating Dictionary from array...")
        with tqdm(total=466544) as pbar:
            for line in arr:
                n_line += 1
                # separate by words by non-alphabetical characters
                words = re.findall(token_pattern, line.lower())
                for word in words:
                    total_word_count += 1
                    if self.create_dictionary_entry(word):
                        unique_word_count += 1
                pbar.update(1)
                if n_line > 100:
                    print(n_line, "test")
                    break

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def create_dictionary(self, fname):
        total_word_count = 0
        unique_word_count = 0
        n_line = 0
        print("Creating Dictionary...")
        with open(fname) as file:
            with tqdm(total=466544) as pbar:
                for line in file:
                    n_line += 1
                    # separate by words by non-alphabetical characters
                    words = re.findall('[a-z]+', line.lower())
                    for word in words:
                        total_word_count += 1
                        if self.create_dictionary_entry(word):
                            unique_word_count += 1
                    pbar.update(1)
                    if n_line == 100:
                        print(n_line, "test")
                        break
        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def get_suggestions(self, string, silent=False):
        """return list of suggested corrections for potentially incorrectly
           spelled word"""
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if self.dictionary[q_item][1] > 0:
                    # word is in dictionary, and is a word from the corpus, and
                    # not already in suggestion list so add to suggestion
                    # dictionary, indexed by the word with value (frequency in
                    # corpus, edit distance)
                    # note q_items that are not the input string are shorter
                    # than input string since only deletes are added (unless
                    # manual dictionary corrections are added)
                    assert len(string) >= len(q_item)
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item))
                    # early exit
                    if (self.verbose < 2) and (len(string) == len(q_item)):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                # the suggested corrections for q_item as stored in
                # dictionary (whether or not q_item itself is a valid word
                # or merely a delete) can be valid corrections
                for sc_item in self.dictionary[q_item][0]:
                    if sc_item not in suggest_dict:

                        # compute edit distance
                        # suggested items should always be longer
                        # (unless manual corrections are added)
                        assert len(sc_item) > len(q_item)

                        # q_items that are not input should be shorter
                        # than original string
                        # (unless manual corrections added)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)

                        # item in suggestions list should not be the same as
                        # the string itself
                        assert sc_item != string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = dameraulevenshtein(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        if (self.verbose < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        # depending on order words are processed, some words
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if verbose
                        # setting not on
                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        # output option 1
        # sort results by ascending order of edit distance and descending
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x:
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction,
        #                                  (frequency in corpus, edit distance)):
        as_list = suggest_dict.items()
        # outlist = sorted(as_list, key=lambda (term, (freq, dist)): (dist, -freq))
        outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

        if self.verbose == 0:
            return outlist[0]
        else:
            return outlist

        '''
        Option 1:
        ['file', 'five', 'fire', 'fine', ...]
        Option 2:
        [('file', (5, 0)),
         ('five', (67, 1)),
         ('fire', (54, 1)),
         ('fine', (17, 1))...]  
        '''

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None


ss = SymSpell(max_edit_distance=2)


def spell_corrector(ss, word_list, words_d) -> str:
    result_list = []
    for word in word_list:
        if word not in words_d:
            suggestion = ss.best_word(word, silent=True)
            if suggestion is not None:
                result_list.append(suggestion)
        else:
            result_list.append(word)

    return " ".join(result_list)


def preprocessing(content, stop_words, ss, words_dict, remove_stopwords=False, correct=False):
    sent = content.lower()
    sent = sent.strip()
    sent = " ".join(normalize_laughting(word_tokenize(sent)))
    sent = normalize_smileys_in_sentence(sent)
    sent = remove_multiple_characters(sent)
    sent = " ".join(word_tokenize(sent))
    sent = remove_accent_before_tokens(sent)
    sent = ''.join([expand_contractions(txt, CONTRACTION_MAP)
                    for txt in sent_tokenize(sent)])
    # sent = replace_numeric(sent)
    if remove_stopwords:
        sent = ' '.join(replace_stopwords(sent, stop_words))
    if correct:
        sent = spell_corrector(ss, word_tokenize(sent), words_dict)
    return sent


if __name__ == '__main__':
    print('Started')
    from nltk.corpus import stopwords
    import spacy

    spacy_nlp = spacy.load('en_core_web_lg')
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    stop_words = set(list(set(stopwords.words('english'))) + list(spacy_stopwords))

    begin = now()
    df = pd.read_csv('data/train_posts.csv', names=['blog', 'class'])
    # file = open('../extracted/preprocessed.csv', "w+")
    n_lines = 0
    with open('data/english_words_479k.txt') as f:
        words = f.readlines()
    eng_words = [word.strip() for word in words]
    print('Creating dictionary for symspell')
    _ = ss.create_dictionary_from_arr(eng_words, token_pattern=r'.+')
    words_dict = {k: 0 for k in eng_words}
    print('Finished dictionary for symspell in', format_delta(begin, now()))
    begin = now()
    for (line, classe) in zip(df['blog'], df['class']):
        # for sentence in nltk.sent_tokenize(line):
        if n_lines % 100 == 0:
            print(str(n_lines) + '/' + str(512629), "lines done in ...", format_delta(begin, now()))
        n_lines += 1
        clean_sentence = ' '.join(nltk.word_tokenize(line)).lower()
        cleaner_sentence = preprocessing(clean_sentence, stopwords, ss, words_dict=words_dict)
        # words = nltk.word_tokenize(cleaner_sentence)
        file.write("{}\t, {}".format(cleaner_sentence, classe) + '\n')

    # preprocessed_corpus = preprocessing(df['blog'])
    runtime = format_delta(begin, now())
    print("finished tokenizing sentences in", runtime)

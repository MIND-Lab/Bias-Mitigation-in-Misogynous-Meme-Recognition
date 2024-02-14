import gc
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import stanza
import spacy_stanza
import string
import re
import torch
from tqdm import tqdm

import pickle

def write_json(data,filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_json(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b


stopwords = ["a", "about", "above", "above", "across", "afterwards", "again", "against",
             "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst",
             "amoungst",
             "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
             "around",
             "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "beforehand",
             "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but",
             "by",
             "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "de", "describe", "detail", "do", "done",
             "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough",
             "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify",
             "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from",
             "front",
             "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "hence", "here", "hereafter",
             "hereby",
             "herein", "hereupon", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into",
             "is",
             "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "meanwhile", "might",
             "mill",
             "more", "moreover", "most", "mostly", "move", "much", "must", "name", "namely", "neither", "never",
             "nevertheless",
             "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "now", "nowhere", "of", "off", "often",
             "on", "once",
             "one", "only", "onto", "or", "other", "others", "otherwise", "out", "over", "part", "per", "perhaps",
             "please",
             "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "should",
             "show",
             "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "sometime", "sometimes",
             "somewhere",
             "still", "such", "system", "take", "ten", "than", "that", "the", "then", "thence", "there", "thereafter",
             "thereby",
             "therefore", "therein", "thereupon", "these", "thick", "thin", "third", "this", "those", "though", "three",
             "through",
             "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty",
             "two", "un",
             "under", "until", "up", "upon", "very", "via", "was", "well", "were", "what", "whatever", "when", "whence",
             "whenever",
             "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which",
             "while", "whither",
             "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet",
             "the",
             "ve", "re", "ll", "10", "11", "18", "oh", "s", "t", "m", "did", "don", "got"]

START_OF_LINE = r"^"
OPTIONAL = "?"
ANYTHING = "."
ZERO_OR_MORE = "*"
ONE_OR_MORE = "+"

SPACE = "\s"
SPACES = SPACE + ONE_OR_MORE
NOT_SPACE = "[^\s]" + ONE_OR_MORE
EVERYTHING_OR_NOTHING = ANYTHING + ZERO_OR_MORE

ERASE = ""
FORWARD_SLASH = "\/"
NEWLINES = r"[\r\n]"


HYPERLINKS = ("http" + "s" + OPTIONAL + ":" + FORWARD_SLASH + FORWARD_SLASH
              + NOT_SPACE + NEWLINES + ZERO_OR_MORE)

def use_preprocessing(df, column):
    """ Compute embeddings through Universal Sentence Encoding algorithm.

    :param df: dataframe
    :param column: column on which apply USE embeddings
    :return: a vector containing computed embeddings
    """
    # Universal Sentence Encoder
    use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    dfs = np.array_split(df, 100)

    # Split in 10 call because during embedding creation an error occurr after 47900 steps.
    text_embeddings = []

    for x in tqdm(dfs):
        np_list = np.asarray(x[column].tolist())
        tensor_list = tf.convert_to_tensor(np_list)
        #print(np_list)

        text_embedding = use(tensor_list)
        text_embeddings = text_embeddings + np.array(text_embedding).tolist()

    del text_embedding
    del dfs
    gc.collect()
    torch.cuda.empty_cache() 

    return text_embeddings
def lemmatization(text, nlp):
    meme = []
    #print(text)
    doc = nlp(text)
    for token in doc:
      #print("TOKEN ", token, " LEMMA ", token.lemma_, " POS ", token.pos_, " TAG ", token.tag_)
      meme.append(token.lemma_)

    review = ' '.join(meme)
    return review 

def text_preprocessing(data, nlp):
    text = re.sub("@[A-Za-z0-9_]+","", data) ##rimouvo menzioni
    text = re.sub(HYPERLINKS, "", text) #url
    text = re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?','',text) ##domini
    text = re.sub(r'\d+', '', text) ### numeri
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) ###punteggiatura
    text = lemmatization(text, nlp)
    #text = NormalizeWithPOS(text) ## lemmatization
    text = re.sub('[^A-Za-z0-9 ]+', '', text) ###char speciali
    text = text.split()
    text = [word.lower() for word in text if not word.lower() in stopwords] ##stopword
    #stem_text = [porter_stemmer.stem(word) for word in text]
    text = ' '.join(text)
    #text = NormalizeWithPOS(text)

    return text


def data_pre_processing(data):
    stanza.download("en")
    nlp = spacy_stanza.load_pipeline("en")
    processed_text = []
    #lemmas = []
    gc.collect()
    for item in data:
        #print(item)
        text = re.sub("@[A-Za-z0-9_]+","", item) ##rimouvo menzioni
        text = re.sub(HYPERLINKS, "", text) #url
        text = re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?','',text) ##domini
        text = re.sub(r'\d+', '', text) ### numeri
        text = text.lower()
        text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) ###punteggiatura
        text = lemmatization(text, nlp)
        #text = NormalizeWithPOS(text) ## lemmatization
        text = re.sub('[^A-Za-z0-9 ]+', '', text) ###char speciali
        text = text.split()
        text = [word.lower() for word in text if not word.lower() in stopwords] ##stopword
        
        #stem_text = [porter_stemmer.stem(word) for word in text]
        #lemmas.append(text)
        text = ' '.join(text)
        #print(text)
        processed_text.append(text)
    #text = NormalizeWithPOS(text)
    return processed_text

def data_pre_processing_masking(data):

    identity_terms = [['demotivational', 'dishwasher', 'promotion', 'whore', 'chick', 'motivate', 'chloroform', 'blond', 'diy', 'belong', "blonde"], ['mcdonald', 'ambulance', 'communism', 'anti', 'valentine', 'developer', 'template', 'weak', 'zipmeme', 'identify']]
    identity_text = identity_terms[0] + identity_terms[1]

    stanza.download("en")
    nlp = spacy_stanza.load_pipeline("en")
    processed_text = []
    #lemmas = []
    gc.collect()
    for item in tqdm(data):
        #print(item)
        text = re.sub("@[A-Za-z0-9_]+","", item) ##rimouvo menzioni
        text = re.sub(HYPERLINKS, "", text) #url
        text = re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?','',text) ##domini
        text = re.sub(r'\d+', '', text) ### numeri
        text = text.lower()
        text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) ###punteggiatura
        text = lemmatization(text, nlp)
        #text = NormalizeWithPOS(text) ## lemmatization
        text = re.sub('[^A-Za-z0-9 ]+', '', text) ###char speciali
        text = text.split()
        text = [word.lower() for word in text if not word.lower() in stopwords] ##stopword
        masked_text = []
        for token in text:
            if token in identity_terms[0]:
                masked_text.append("[POS-MASK]")
            elif token in identity_terms[1]:
                masked_text.append("[NEG-MASK]")
            else:
                masked_text.append(token)
                
        #stem_text = [porter_stemmer.stem(word) for word in text]
        #lemmas.append(text)
        text = ' '.join(masked_text)
        #print(text)
        processed_text.append(text)
    #text = NormalizeWithPOS(text)
    return processed_text

def data_pre_processing_censored(data):
    identity_terms = [['demotivational', 'dishwasher', 'promotion', 'whore', 'chick', 'motivate', 'chloroform', 'blond', 'diy', 'belong', "blonde"], ['mcdonald', 'ambulance', 'communism', 'anti', 'valentine', 'developer', 'template', 'weak', 'zipmeme', 'identify']]
    identity_text = identity_terms[0] + identity_terms[1]
    stanza.download("en")
    nlp = spacy_stanza.load_pipeline("en")
    processed_text = []
    #lemmas = []
    gc.collect()
    for item in tqdm(data):
        #print(item)
        text = re.sub("@[A-Za-z0-9_]+","", item) ##rimouvo menzioni
        text = re.sub(HYPERLINKS, "", text) #url
        text = re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?','',text) ##domini
        text = re.sub(r'\d+', '', text) ### numeri
        text = text.lower()
        text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) ###punteggiatura
        text = lemmatization(text, nlp)
        #text = NormalizeWithPOS(text) ## lemmatization
        text = re.sub('[^A-Za-z0-9 ]+', '', text) ###char speciali
        text = text.split()
        text = [word.lower() for word in text if not word.lower() in stopwords] ##stopword
        censored = []
        for token in text:
            if token not in identity_text:
                censored.append(token)
        #stem_text = [porter_stemmer.stem(word) for word in text]
        #lemmas.append(text)
        text = ' '.join(censored)
        #print(text)
        processed_text.append(text)
    #text = NormalizeWithPOS(text)
    return processed_text

def apply_lemmatization_stanza(texts):
    """ Apply lemmatizaion with post tagging operatio through Stanza.
    Remove stopwords and puntuation.
    Lower case """
    stanza.download("en")
    nlp = spacy_stanza.load_pipeline("en")
    processed_text = []

    for text in texts:
        text = re.sub("@[A-Za-z0-9_]+", "", text)  # mentions
        text = re.sub(HYPERLINKS, "", text)  # url
        text = re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?', '',
                      text)  # domains
        text = re.sub(r'\d+', '', text)  # numbers
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))  ###punteggiatura
        text = re.sub('[^A-Za-z0-9 ]+', '', text)  # special char
        text = text.lower()

        meme = []
        doc = nlp(text)
        for token in doc:
            meme.append(token.lemma_)
        text = ' '.join(meme)

        text = text.split()
        text = [word for word in text if not word.lower() in stopwords]  # stopwords
        text = ' '.join(text)
        processed_text.append(text)
    return processed_text

def clear_text_lemma(testo, nlp):
    stopwords = ["a", "about", "above", "above", "across", "afterwards", "again", "against",
                 "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst",
                 "amoungst",
                 "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
                 "around",
                 "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "beforehand",
                 "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but",
                 "by",
                 "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "de", "describe", "detail", "do", "done",
                 "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough",
                 "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify",
                 "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from",
                 "front",
                 "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "hence", "here", "hereafter",
                 "hereby",
                 "herein", "hereupon", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into",
                 "is",
                 "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "meanwhile", "might",
                 "mill",
                 "more", "moreover", "most", "mostly", "move", "much", "must", "name", "namely", "neither", "never",
                 "nevertheless",
                 "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "now", "nowhere", "of", "off", "often",
                 "on", "once",
                 "one", "only", "onto", "or", "other", "others", "otherwise", "out", "over", "part", "per", "perhaps",
                 "please",
                 "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "should",
                 "show",
                 "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "sometime", "sometimes",
                 "somewhere",
                 "still", "such", "system", "take", "ten", "than", "that", "the", "then", "thence", "there", "thereafter",
                 "thereby",
                 "therefore", "therein", "thereupon", "these", "thick", "thin", "third", "this", "those", "though", "three",
                 "through",
                 "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty",
                 "two", "un",
                 "under", "until", "up", "upon", "very", "via", "was", "well", "were", "what", "whatever", "when", "whence",
                 "whenever",
                 "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which",
                 "while", "whither",
                 "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet",
                 "the",
                "ve", "re", "ll", "10", "11", "18", "oh", "s", "t", "m", "did", "don", "got"]
    """
    Remove punctuation, brings to lowercase, remove special char, apply Stanza lemmatization
    :param testo: text to process
    :return: processed text
    """

    rev = []

    testo = testo.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    testo = testo.lower()
    testo = re.sub(r'\d+', '', testo)
    testo = re.sub('[^A-Za-z0-9 ]+', '', testo)
    testo = " ".join(testo.split())  # single_spaces

    doc = nlp(testo)
    for token in doc:
        rev.append(token.lemma_)

    for word in list(rev):  # iterating on a copy since removing will mess things up
        if word in stopwords:
            rev.remove(word)
    return rev



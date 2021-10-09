import numpy as np
import nltk
import re
# import math
import gensim
from gensim import corpora
from gensim.parsing.preprocessing import remove_stopwords
from scipy import spatial
from nltk.corpus import brown





def clean_sentence(sentence, stopwords=False):
    sentence = sentence.lower().strip()
    if stopwords:
        sentence = remove_stopwords(sentence)
    return sentence

def get_cleaned_sentences(tokens,question,model, stopwords=False):
    if True:
        question = remove_stopwords(question)
    cleaned_sentences = []
    for line in tokens:
        cleaned = clean_sentence(line, stopwords)
        cleaned_sentences.append(cleaned)

    most_sim = ""
    sim = 1
    for line in cleaned_sentences:
        if True:
            line1 = re.sub(r'[^a-z0-9s]', '', line)
            # line1 = remove_stopwords(line1)
        temp = model.wmdistance(line1,question)
        if temp<sim and temp<0.6:
            most_sim = line
            sim = temp
    if most_sim == "":
        print("No Answer in the tree defined")
    else:
        print(most_sim)
    return cleaned_sentences





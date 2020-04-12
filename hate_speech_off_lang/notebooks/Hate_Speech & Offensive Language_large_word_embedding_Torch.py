from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
import scipy
from scipy import stats

import os

cwd_nb = os.getcwd()

cwd = os.path.dirname(cwd_nb)

train = pd.read_csv(cwd + '/data/raw/train.tsv', sep='\t', encoding='latin_1', header=None)
train = train.dropna()
train.columns = ['hate_speech', 'off_lang', 'text']

# Word-Embeddings aus SpaCy - kleines Model
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz --user
# !pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.2.5/en_core_web_lg-2.2.5.tar.gz --user
import en_core_web_lg

nlp = en_core_web_lg.load()

from tqdm import tqdm

features = []
labels_off = []
labels_hate = []

for i in tqdm(range(0, len(train))):
    features.append(nlp(train.text[i]).vector)
    labels_off.append(train.off_lang[i])
    labels_hate.append(train.hate_speech[i])

features_lg = np.array(features)
labels_off = np.array(labels_off)
labels_hate = np.array(labels_hate)

print(features_lg.shape)

# Test auf Goldstandard

test = pd.read_csv(cwd + '/data/raw/test.tsv', sep='\t', encoding='latin_1', header=None)
test.columns = ['hate_speech', 'off_lang', 'text']
from tqdm import tqdm

features = []
labels_test_lang = []
labels_test_hate = []

for i in tqdm(range(0, len(test))):
    features.append(nlp(test.text[i]).vector)
    labels_test_lang.append(test.off_lang[i])
    labels_test_hate.append(test.hate_speech[i])

features_test_lg = np.array(features)
labels_test_gs_lang_lg = np.array(labels_test_lang)
labels_test_gs_hate_lg = np.array(labels_test_hate)

###### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Training Model
###### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


###### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
##Testing
###### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
feat_lg = features_test_lg.reshape(-1, features_test_lg.shape[1])
pred_hate_test_lg = model_hate_speech_MLP_lg.predict(feat_lg).ravel()
pred_off_test_lg = model_off_language_MLP_lg.predict(feat_lg).ravel()

print(scipy.stats.pearsonr(pred_hate_test_lg, labels_test_gs_hate_lg))

print(scipy.stats.pearsonr(pred_off_test_lg, labels_test_gs_lang_lg))

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


import spacy
import torch


is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

try:
    nlp = spacy.load("en_trf_bertbaseuncased_lg")
except:
    import en_trf_bertbaseuncased_lg
    nlp = en_trf_bertbaseuncased_lg.load()

from tqdm import tqdm

features = []
labels_off = []
labels_hate = []

for i in tqdm(range(0, len(train[:300]))):
    features.append(nlp(train.text[i]).vector)
    labels_off.append(train.off_lang[i])
    labels_hate.append(train.hate_speech[i])

features_lg = np.array(features)
labels_off = np.array(labels_off)
labels_hate = np.array(labels_hate)

print(features_lg.shape)

from tensorflow import keras

print(keras.__version__)

import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# let's see what compute devices we have available, hopefully a GPU
# sess = tf.Session()
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import pickle

# saving
with open('tokenizer_en_trf_bertbaseuncased_lg.pickle', 'wb') as handle:
    pickle.dump(en_trf_bertbaseuncased_lg, handle, protocol=pickle.HIGHEST_PROTOCOL)



###### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
## MOdel für Hate-Speech
###### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model_hate_speech_MLP_lg = Sequential()
model_hate_speech_MLP_lg.add(Dense(768, input_dim=768, kernel_initializer='normal', activation='relu'))
model_hate_speech_MLP_lg.add(Dense(300, kernel_initializer='normal', activation='relu'))
model_hate_speech_MLP_lg.add(Dense(50, kernel_initializer='normal', activation='relu'))
model_hate_speech_MLP_lg.add(Dense(1, kernel_initializer='normal'))
print(model_hate_speech_MLP_lg.summary())

model_hate_speech_MLP_lg.compile(optimizer='adam', loss='mean_squared_error')
from time import time

start = time()
history_hate = model_hate_speech_MLP_lg.fit(features_lg, labels_hate, epochs=100, batch_size=64, verbose=1)
end = time()

model_hate_speech_MLP_lg.save(cwd + '/models/hate_speech_model_BERT.h5')

###### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
## Model für Off_Language
###### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###



model_off_language_MLP_lg = Sequential()
model_off_language_MLP_lg.add(Dense(768, input_dim=768, kernel_initializer='normal', activation='relu'))
model_off_language_MLP_lg.add(Dense(300, kernel_initializer='normal', activation='relu'))
model_off_language_MLP_lg.add(Dense(50, kernel_initializer='normal', activation='relu'))
model_off_language_MLP_lg.add(Dense(1, kernel_initializer='normal'))

model_off_language_MLP_lg.compile(optimizer='adam', loss='mean_squared_error')

from time import time

start = time()
history_off = model_off_language_MLP_lg.fit(features_lg, labels_off, epochs=100, batch_size=64, verbose=1)
end = time()

model_off_language_MLP_lg.save(cwd + '/models/offensive_language_model_BERT.h5')

###### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Test auf Goldstandard
###### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

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

feat_lg = features_test_lg.reshape(-1, features_test_lg.shape[1])
pred_hate_test_lg = model_hate_speech_MLP_lg.predict(feat_lg).ravel()
pred_off_test_lg = model_off_language_MLP_lg.predict(feat_lg).ravel()

print(scipy.stats.pearsonr(pred_hate_test_lg, labels_test_gs_hate_lg))

print(scipy.stats.pearsonr(pred_off_test_lg, labels_test_gs_lang_lg))

from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import pandas as pd
import numpy as np

import os

cwd_nb = os.getcwd()
print(cwd_nb)

cwd = os.path.dirname(cwd_nb)
print(cwd)

train = pd.read_csv(cwd + '/data/raw/train.tsv', sep='\t', encoding='latin_1', header=None)
train = train.dropna()
train.columns = ['hate_speech', 'off_lang', 'text']

# Word-Embeddings aus SpaCy - kleines Model
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz --user
import en_core_web_sm

nlp = en_core_web_sm.load()

from tqdm import tqdm

features = []
labels_off = []
labels_hate = []

for i in tqdm(range(0, len(train))):
    features.append(nlp(train.text[i]).vector)
    labels_off.append(train.off_lang[i])
    labels_hate.append(train.hate_speech[i])

features = np.array(features)
labels_off = np.array(labels_off)
labels_hate = np.array(labels_hate)

print(features.shape)

features_train = features[:13217]
features_test = features[13217:]
labels_train_off = labels_off[:13217]
labels_test_off = labels_off[13217:]
labels_train_hate = labels_hate[:13217]
labels_test_hate = labels_hate[13217:]

features_train = np.reshape(features_train, (features_train.shape[0], features_train.shape[1], 1))

from tensorflow import keras

print(keras.__version__)

import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.__version__)

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# let's see what compute devices we have available, hopefully a GPU
sess = tf.Session()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

## LSTM für Hate-Speech
model_hate_speech = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=50, return_sequences=True, recurrent_dropout=0.2, input_shape=(96, 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=50, return_sequences=True, recurrent_dropout=0.2, ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=50),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=1)
])

print(model_hate_speech.summary())

from tensorflow.python.client import device_lib

device_lib.list_local_devices()

model_hate_speech.compile(optimizer='adam', loss='mean_squared_error')

from time import time

start = time()
history2 = model_hate_speech.fit(features_train, labels_train_hate, epochs=10, batch_size=32, verbose=1)
end = time()

model_hate_speech.save(cwd + '/models/hate_speech_model.h5')

features_test = np.reshape(features_test, (features_test.shape[0], features_test.shape[1], 1))
pred_hate = model_hate_speech.predict(features_test).ravel()
import scipy
from scipy import stats

print(scipy.stats.pearsonr(pred_hate, labels_test_hate))

## LSTM für Off_Language
model_off_language = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=50, return_sequences=True, recurrent_dropout=0.2, input_shape=(96, 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=50, return_sequences=True, recurrent_dropout=0.2, ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=50),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=1)
])

model_off_language.compile(optimizer='adam', loss='mean_squared_error')

from time import time

start = time()
history = model_off_language.fit(features_train, labels_train_off, epochs=10, batch_size=32, verbose=1)
end = time()

model_off_language.save(cwd + '/models/offensive_language_model.h5')

pred_off_model = model_off_language.predict(features_test).ravel()
print(scipy.stats.pearsonr(pred_off_model, labels_test_off))

# Test auf Goldstandard
test = pd.read_csv(cwd + '/data/raw/test.tsv', sep='\t', encoding='latin_1', header=None)
test.columns = ['hate_speech', 'off_lang', 'text']
from tqdm import tqdm

features = []
labels_gs_off = []
labels_gs_hate = []

for i in tqdm(range(0, len(test))):
    features.append(nlp(test.text[i]).vector)
    labels_gs_off.append(test.off_lang[i])
    labels_gs_hate.append(test.hate_speech[i])

features_test = np.array(features)
labels_gs_off = np.array(labels_gs_off)
labels_gs_hate = np.array(labels_gs_hate)

features_test = np.reshape(features_test, (features_test.shape[0], features_test.shape[1], 1))
pred_hate_test = model_hate_speech.predict(features_test).ravel()
pred_off_test = model_off_language.predict(features_test).ravel()

print(scipy.stats.pearsonr(pred_off_test, labels_gs_off))

print(scipy.stats.pearsonr(pred_hate_test, labels_gs_hate))

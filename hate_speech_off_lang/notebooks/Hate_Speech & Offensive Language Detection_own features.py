from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import pandas as pd
import numpy as np

# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz --user
import en_core_web_sm

nlp = en_core_web_sm.load()

## Für Hate-Speech:
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords

# stop = stopwords.words('english')
# stop.append("i'm")
# df_train = pd.read_csv(r'C:\Users\simon\Desktop\Bildung\Master Data Science\5.Semester\50200 Webmining\Präsenz\train.tsv', sep = '\t', encoding = 'latin_1', header=None)
# df_train.rename(columns = {0:'hate_speech',1:'off_lang',2:'text'},inplace = True)
# df_train['text'] = df_train['text'].str.lower().str.replace(r'[\?(),;!"“]', '')
# df_train['text'] =  df_train['text'].str.replace(r'@[a-z0-9\-_]+:? ?', '').str.replace(r'^rt ', '').str.replace(r' rt', '').str.replace(r'&#[0-9]+;', '').str.replace(r'&#[0-9]+', '').str.replace(r'https?://[a-z\/\.0-9]+', '').str.replace(r'[#\.]','')
# df_train['text_split'] = df_train['text'].str.split(' ')
# df_train['text_split'] = df_train['text_split'].apply(lambda x: [item for item in x if item not in stop])
# df_train['text']  = df_train['text_split'].apply(' '.join)


# from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords_en
# spacy_stopwords_en.add("PRON")

# from sklearn.feature_extraction.text import TfidfVectorizer
# learn_tfidf = TfidfVectorizer(max_features=20000, stop_words=spacy_stopwords_en)
# learn_vectors = learn_tfidf.fit_transform(df_train.text_split.apply(str))


#
# X = df_train['text']
# Y = df_train.hate_speech.values
#
# from sklearn import model_selection
# seed = 42
# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=seed)
#
# df = pd.DataFrame()
#
# doc = nlp(df_train.text[0])
#
# #from spacy import displacy
# #displacy.serve(doc, style="dep")
#
# df_train = df_train.dropna()

train = pd.read_csv(r'C:\Users\simon\Desktop\Bildung\Master Data Science\5.Semester\50200 Webmining\Präsenz\train.tsv',
                    sep='\t', encoding='latin_1', header=None)
train = train.dropna()
train.columns = ['hate_speech', 'off_lang', 'text']

from tqdm import tqdm

features = []
labels = []
labels2 = []

for i in tqdm(range(0, len(train))):
    features.append(nlp(train.text[i]).vector)
    labels.append(train.off_lang[i])
    labels2.append(train.hate_speech[i])

features = np.array(features)
labels = np.array(labels)
labels2 = np.array(labels2)

print(features.shape)

features_train = features[:13217]
features_test = features[13217:]
labels_train = labels[:13217]
labels_test = labels[13217:]
labels_train2 = labels2[:13217]
labels_test2 = labels2[13217:]

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
history2 = model_hate_speech.fit(features_train, labels_train2, epochs=10, batch_size=32, verbose=1)
end = time()

pred_off = model_hate_speech.predict(features_test).ravel()
pred_hate = model_hate_speech.predict(features_test).ravel()

import scipy

print(scipy.stats.pearsonr(pred_hate, labels_test2))

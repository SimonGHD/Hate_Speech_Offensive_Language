from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd

### Switch environment

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import os

cwd_nb = os.getcwd()

cwd = os.path.dirname(cwd_nb)

# Import the model

test = pd.read_csv(cwd + '/data/raw/test.tsv', sep='\t', encoding='latin_1', header=None)

from keras.models import load_model

# load model
model_off_language_MLP_lg = load_model(cwd + '/models/offensive_language_model_lg_word_embed.h5')

print('Model loaded successfully')

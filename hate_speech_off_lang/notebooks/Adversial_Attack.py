from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import os

cwd_nb = os.getcwd()

cwd = os.path.dirname(cwd_nb)

# Import the model

test = pd.read_csv(cwd + '/data/raw/test.tsv', sep='\t', encoding='latin_1', header=None)

from tensorflow import keras

model_off_language_MLP_lg = keras.models.load_model(cwd + '/models/offensive_language_model_BERT.h5')

print('Model loaded successfully')

# from textattack.transformations import WordSwap
#
# class BananaWordSwap(WordSwap):
#     """ Transforms an input by replacing any word with 'banana'.
#     """
#
#     # We don't need a constructor, since our class doesn't require any parameters.
#
#     def _get_replacement_words(self, word):
#         """ Returns 'banana', no matter what 'word' was originally.
#
#             Returns a list with one item, since `_get_replacement_words` is intended to
#                 return a list of candidate replacement words.
#         """
#         return ['banana']
#
#
# import transformers
# from textattack.models.tokenizers import AutoTokenizer
#
# model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
# model.tokenizer = AutoTokenizer("textattack/bert-base-uncased-ag-news")
#
# # Create the goal function using the model
# from textattack.goal_functions import UntargetedClassification
# goal_function = UntargetedClassification(model)
#
# # Import the dataset
# from textattack.datasets import HuggingFaceNlpDataset
# dataset = HuggingFaceNlpDataset("ag_news", None, "test")

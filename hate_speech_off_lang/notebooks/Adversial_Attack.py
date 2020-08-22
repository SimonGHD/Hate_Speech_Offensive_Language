from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd

### Switch environment

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import os

cwd_nb = os.getcwd()

cwd = os.path.dirname(cwd_nb)


import pyarrow

print(pyarrow.__version__)

# n TextAttack, many transformations involve word swaps: they take a word and try and find suitable substitutes.
from textattack.transformations import WordSwap


# In TextAttack, there’s an abstract WordSwap class that handles the heavy lifting of breaking sentences into words and avoiding replacement of stopwords.

class LoveWordSwap(WordSwap):
    """ Transforms an input by replacing any word with 'banana'.
    """

    # We don't need a constructor, since our class doesn't require any parameters.

    def _get_replacement_words(self, word):
        """ Returns 'banana', no matter what 'word' was originally.

            Returns a list with one item, since `_get_replacement_words` is intended to
                return a list of candidate replacement words.
        """
        return ['love']


# We can extend WordSwap and implement a single method, _get_replacement_words, to indicate to replace each word with ‘banana’.


# Now we have the transformation chosen, but we’re missing a few other things.
# To complete the attack, we need to choose the search method and constraints.
# And to use the attack, we need a goal function, a model and a dataset.
# (The goal function indicates the task our model performs – in this case, classification – and the type of attack –
# in this case, we’ll perform an untargeted attack.)

import transformers
from textattack.models.tokenizers import AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
tokenizer = AutoTokenizer("textattack/bert-base-uncased-ag-news")

model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
# Create the goal function using the model
from textattack.goal_functions import UntargetedClassification

goal_function = UntargetedClassification(model_wrapper)
#

# Import the dataset

# Greedy Search attack
from textattack.search_methods import GreedySearch
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.shared import Attack
from textattack.loggers import FileLogger  # tracks a dataframe for us.
from textattack.attack_results import SuccessfulAttackResult

# We're going to use our Banana word swap class as the attack transformation.
transformation = LoveWordSwap()
# We'll constrain modification of already modified indices and stopwords
constraints = [RepeatModification(),
               StopwordModification()]
# We'll use the Greedy search method
search_method = GreedySearch()
# Now, let's make the attack from the 4 components:
attack = Attack(goal_function, constraints, transformation, search_method)

print(attack)

from textattack.datasets import HuggingFaceNlpDataset

dataset = HuggingFaceNlpDataset("ag_news", None, "test")

print(dataset)
# results_iterable = attack.attack_dataset(dataset)
# #
#
# logger = FileLogger(filename=cwd + '/data/processed/attack_logger.csv')
#
# num_runs = 10
#
# num_successes = 0
# while num_successes < num_runs:
#     result = next(results_iterable)
#     if isinstance(result, SuccessfulAttackResult):
#         logger.log_attack_result(result)
#         num_successes += 1
#         print(f'{num_successes} of {num_runs} successes complete.')
#
#
# pd.options.display.max_colwidth = 480  # increase colum width so we can actually read the examples

# from IPython.core.display import display, HTML

# print(display(HTML(logger.df[['original_text', 'perturbed_text']].to_html(escape=False))))


# # Attacking Custom Samples
#
# # For AG News, labels are 0: World, 1: Sports, 2: Business, 3: Sci/Tech
#
# custom_dataset = [
#     ('Malaria deaths in Africa fall by 5% from last year', 0),
#     ('Washington Nationals defeat the Houston Astros to win the World Series', 1),
#     ('Exxon Mobil hires a new CEO', 2),
#     ('Microsoft invests $1 billion in OpenAI', 3),
# ]
#
# results_iterable = attack.attack_dataset(custom_dataset)
#
# logger = CSVLogger(color_method='html')
#
# for result in results_iterable:
#     logger.log_attack_result(result)
#
# display(HTML(logger.df[['original_text', 'perturbed_text']].to_html(escape=False)))
#
# mydoc = docx.Document()
# mydoc.add_paragraph(logger.df[['original_text', 'perturbed_text']])
# mydoc.save(cwd + '/data/Beispiel_extra.docx')


#######################################################################
#### ENDE Beispiele
#######################################################################

# Import the model
test = pd.read_csv(cwd + '/data/raw/test.tsv', sep='\t', encoding='latin_1', header=None)
test.columns = ['hate_speech', 'off_lang', 'text']

cols = ['text', 'off_lang']
test = test[cols]
test_array = test.to_numpy()
# print(test_array)
from tensorflow.keras.models import load_model

model_off_language_MLP_lg = load_model(cwd + '/models/offensive_language_model_lg_word_embed.h5')

import en_core_web_lg
import spacy

try:
    nlp = en_core_web_lg.load()
except:
    nlp = spacy.load('en_core_web_lg')

features = []
labels_test_lang = []
labels_test_hate = []

# load model

# model_off_language_MLP_lg.tokenizer = spacy.load("en_trf_bertbaseuncased_lg") --> macht aktuell Probleme: cant set tokenizer
# model_off_language_MLP_lg.tokenizer = nlp

tokenizer = nlp

import numpy as np

from textattack.models.wrappers import ModelWrapper


class CustomModelWrapper(ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, text_input_list):
        features = []
        for i in range(0, len(text_input_list)):
            features.append(nlp(text_input_list[i]).vector)
        features_test_lg = np.array(features)

        feat_lg = features_test_lg.reshape(-1, features_test_lg.shape[1])
        pred_lang_test_lg = self.model.predict(feat_lg).ravel()
        return pred_lang_test_lg


model_wrapper = CustomModelWrapper(model_off_language_MLP_lg)

print(CustomModelWrapper(model_off_language_MLP_lg)(['I hate you so much', 'I love u ']))

#
print('Model loaded successfully')

from textattack.goal_functions import TargetedClassification

goal_function = TargetedClassification(model_wrapper, target_class=3)
# goal_function =  TargetedClassification(model_off_language_MLP_lg)


# We're going to use our Banana word swap class as the attack transformation.
transformation = LoveWordSwap()
# We'll constrain modification of already modified indices and stopwords
constraints = [RepeatModification(),
               StopwordModification()]
# We'll use the Greedy search method
search_method = GreedySearch()
# Now, let's make the attack from the 4 components:
attack = Attack(goal_function, constraints, transformation, search_method)

print(attack)

results_iterable = attack.attack_dataset(test_array)

logger = FileLogger(filename=cwd + '/data/processed/attack_logger_Hatespeech_Bert.csv')

num_runs = 10

num_successes = 0
while num_successes < num_runs:
    result = next(results_iterable)
    print(result)
    if isinstance(result, SuccessfulAttackResult):
        logger.log_attack_result(result)
        num_successes += 1
        print(f'{num_successes} of {num_runs} successes complete.')

import pandas as pd

pd.options.display.max_colwidth = 480

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

from tensorflow.keras.models import load_model

# load model
model_off_language_MLP_lg = load_model(cwd + '/models/offensive_language_model_lg_word_embed.h5')

print('Model loaded successfully')
import pyarrow

print(pyarrow.__version__)

# n TextAttack, many transformations involve word swaps: they take a word and try and find suitable substitutes.
from textattack.transformations import WordSwap


# In TextAttack, there’s an abstract WordSwap class that handles the heavy lifting of breaking sentences into words and avoiding replacement of stopwords.

class BananaWordSwap(WordSwap):
    """ Transforms an input by replacing any word with 'banana'.
    """

    # We don't need a constructor, since our class doesn't require any parameters.

    def _get_replacement_words(self, word):
        """ Returns 'banana', no matter what 'word' was originally.

            Returns a list with one item, since `_get_replacement_words` is intended to
                return a list of candidate replacement words.
        """
        return ['banana']


# We can extend WordSwap and implement a single method, _get_replacement_words, to indicate to replace each word with ‘banana’.


# Now we have the transformation chosen, but we’re missing a few other things.
# To complete the attack, we need to choose the search method and constraints.
# And to use the attack, we need a goal function, a model and a dataset.
# (The goal function indicates the task our model performs – in this case, classification – and the type of attack –
# in this case, we’ll perform an untargeted attack.)

import transformers
from textattack.models.tokenizers import AutoTokenizer

model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
model.tokenizer = AutoTokenizer("textattack/bert-base-uncased-ag-news")

# Create the goal function using the model
from textattack.goal_functions import UntargetedClassification

# goal_function = UntargetedClassification(model)
goal_function = UntargetedClassification(model_off_language_MLP_lg)

# Import the dataset
from textattack.datasets import HuggingFaceNlpDataset

dataset = HuggingFaceNlpDataset("ag_news", None, "test")

# Greedy Search attack
from textattack.search_methods import GreedySearch
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.shared import Attack

# We're going to use our Banana word swap class as the attack transformation.
transformation = BananaWordSwap()
# We'll constrain modification of already modified indices and stopwords
constraints = [RepeatModification(),
               StopwordModification()]
# We'll use the Greedy search method
search_method = GreedySearch()
# Now, let's make the attack from the 4 components:
attack = Attack(goal_function, constraints, transformation, search_method)

print(attack)

from textattack.loggers import CSVLogger  # tracks a dataframe for us.
from textattack.attack_results import SuccessfulAttackResult

# results_iterable = attack.attack_dataset(dataset)
results_iterable = attack.attack_dataset(test)

logger = CSVLogger(color_method='html')

num_successes = 0
while num_successes < 10:
    result = next(results_iterable)
    if isinstance(result, SuccessfulAttackResult):
        logger.log_attack_result(result)
        num_successes += 1
        print(f'{num_successes} of 10 successes complete.')

import pandas as pd

pd.options.display.max_colwidth = 480  # increase colum width so we can actually read the examples

from IPython.core.display import display, HTML

display(HTML(logger.df[['original_text', 'perturbed_text']].to_html(escape=False)))

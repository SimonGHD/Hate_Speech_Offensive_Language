import en_core_web_lg

nlp = en_core_web_lg.load()
import numpy as np
from tqdm import tqdm

def make_features(train):
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

    return features_lg, labels_off, labels_hate


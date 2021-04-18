import random

import pandas as pd
import numpy as np
from feature_maker import FeatureCreation
import spacy
from random import sample
import random

# load data
train = pd.read_csv("train_E6oV3lV.csv")
test = pd.read_csv("test_tweets_anuFYb8.csv")
# create features
"""train_features = FeatureCreation(train,"tweet").feature_chain()
test_features = FeatureCreation(test,"tweet").feature_chain()

print(train_features.head())"""

nlp = spacy.load('en_core_web_lg')
# Disabling other pipes because we don't need them and it'll speed up this part a bit
with nlp.disable_pipes():
    doc_vectors = np.array([nlp(text).vector for text in train.tweet])
doc_vectors.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(doc_vectors,
                                                    train.tweet,
                                                    test_size=0.15,
                                                    random_state=1)
from sklearn.svm import LinearSVC
svc = LinearSVC(random_state=1, dual=False, max_iter=1000)
svc.fit(X_train, y_train)
print(f"Accuracy: {svc.score(X_test, y_test) * 100:.3f}%", )

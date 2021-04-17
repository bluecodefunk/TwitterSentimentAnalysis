import pandas as pd
import numpy as np
from feature_maker import FeatureCreation

# load data
train = pd.read_csv("train_E6oV3lV.csv")
test = pd.read_csv("test_tweets_anuFYb8.csv")
# create features
train_features = FeatureCreation(train,"tweet").feature_chain()
test_features = FeatureCreation(test,"tweet").feature_chain()

train_features.head()
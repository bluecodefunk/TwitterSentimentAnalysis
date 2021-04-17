import pandas as pd
import numpy as np
import string
import re
import gc
from collections import defaultdict
from wordcloud import STOPWORDS


class FeatureCreation:
    def __init__(self, dataset, textcol):
        self.dataset = dataset
        self.textcol = textcol

    def feature_chain(self):
        df_var = self.dataset
        # word count
        text_col = self.textcol
        df_var['word_count'] = df_var[text_col].apply(lambda x: len(str(x).split()))
        # get unique words
        df_var['unique_word_count'] = df_var[text_col].apply(lambda x:
                                                             len(set(str(x).split())))
        # get stop words
        df_var['stop_word_count'] = df_var[text_col].apply(lambda x: len([
            w for w in str(x).lower().split() if w in STOPWORDS
             ]))
        df_var['url_count'] = df_var[text_col].apply(lambda x: len([
            w for w in str(x).lower().split() if "http" in w or "https" in w
        ]))
        df_var['mean_word_length'] = df_var[text_col].apply(lambda x: np.mean([
            len(w) for w in str(x).split()
        ]))
        df_var['char_count'] = df_var[text_col].apply(lambda x: len(str(x)))
        df_var['punctuation_count'] = df_var[text_col].apply(lambda x: len([
            w for w in str(x) if w in string.punctuation
        ]))
        df_var['hashtag_count'] = df_var[text_col].apply(lambda x: len([
            w for w in str(x) if w == "#"
        ]))
        df_var['mention_count'] = df_var[text_col].apply(lambda x: len([
            w for w in str(x) if w == "@"
        ]))

        return df_var

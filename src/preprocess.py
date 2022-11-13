import os
import config
import pandas as pd 

class Preprocess:
    def __init__(self):
        self.train_df = pd.read_csv(config.TRAIN_PATH)
        self.cols = ['text', 'label']
        self.train_df.drop([col for col in self.train_df.columns if \
                                col not in self.cols], axis=1, inplace=True)

        #print(self.train_df.columns)
        self.text = self.train_df.text
        self.label = self.train_df.label



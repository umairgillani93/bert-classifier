import os
import config
import dataset
import utils
import torch
import pandas as pd 
import numpy as np 
import torch.nn as nn

from model import BERTModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

# Instantiate the dataset object
bd = dataset.BertDataset()

def run():
    dfx = pd.read_csv(config.TRAIN_PATH).fillna('none')
    cols = ['text', 'label']
    dfx.drop([col for col in dfx.columns if \
                            col not in cols], axis=1, inplace=True)

    print(dfx.head())
    df_train, df_valid = train_test_split(
            dfx, test_size=0.3,
            random_state=42,
            stratify=dfx.label.values
            )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # create the train dataset
    train_dataset = bd._get

    

if __name__ == '__main__':
    run()
    


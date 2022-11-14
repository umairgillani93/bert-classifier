# Created by Syed Umair Gillani; Umairgillani93@gmail.com
import os
import config
import dataset
import utils
import torch
import pandas as pd 
import numpy as np 
import torch.nn as nn
from preprocess import Preprocess
from model import BERTModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run():
    dfx = pd.read_csv(config.TRAIN_PATH).fillna('none')
    cols = ['text', 'label']
    dfx.drop([col for col in dfx.columns if \
                            col not in cols], axis=1, inplace=True)

    #print(dfx.head())
    df_train, df_valid = train_test_split(
            dfx, test_size=0.3,
            random_state=42,
            stratify=dfx.label.values
            )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # create Preprocess object
    pp = Preprocess()
    print(f'\nPreprocess object created')

    # training dataset
    train_dataset = dataset.BertDataset(
            text = pp.text.values,
            label = pp.label.values)
    
    #  validation dataset
    train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
            num_workers=4)
    
    
    # set the device 
    device = torch.device(config.DEVICE)

    # initiate model
    model = BERTModel()
    model.to(device)

    param_optimizer = list(model.named_parameters())

    #print(f'model parameters: {param_optimizer}')

    # no decay for the weights updation
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optim_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.0}
            ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    print(f'\nTraining steps: {num_train_steps}')
    optim = torch.optim.AdamW(model.parameters(), lr=3e-5)
    schedular = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=0, num_training_steps = num_train_steps)

    best_acc = 0
    for epoch in range(config.EPOCHS):
        print(' >> Trainnig model .. ')
        utils.train_fn(train_data_loader, model, optim, device, schedular)
        outputs, targets = engine.eval_fn(train_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f'Accuracy score: {accuracy}')
        
        if accuracy > best_acc:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_acc = accuracy


if __name__ == '__main__':
    run()


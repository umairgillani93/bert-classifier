import os
import sys
import torch
import config
import pandas as pd 
from preprocess import Preprocess

class BertDataset:
    def __init__(self, text, label):
        #self.train_df = pd.read_csv(config.TRAIN_PATH)
        #self.cols = ['text', 'label']
        #self.train_df.drop([col for col in self.train_df.columns if \
        #                        col not in self.cols], axis=1, inplace=True)

        ##print(self.train_df.columns)
        #self.text = self.train_df.text
        #self.label = self.train_df.label
        self.text = text
        self.label = label
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        #print('everything done')


    def _getdata(self):
        '''
        return the data in required format
        to feed to NN later
        '''
        return self.text.values, self.label.values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = str(self.text[idx])
        text = " ".join(text.split())
        
        # inputs from tokenizer
        inputs = self.tokenizer.encode_plus(text,
                                            None,
                                            max_length=self.max_len,
                                            pad_to_max_length=True)

        #print(f'\nType: {type(inputs)}')
        #print(f'\ninput keys: {inputs.keys()}')

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        # conver all the above parameters to tensors and return as a dictionary
        return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.label[idx], dtype=torch.float) } 

#if __name__ == '__main__':
#    bd = BertDataset(text_, label_)
#    print(bd.__getitem__(2))
#


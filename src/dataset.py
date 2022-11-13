import os
import sys
import torch
import config
import pandas as pd 

class BertDataset:
    def __init__(self):
        self.train_df = pd.read_csv(config.TRAIN_PATH)
        self.cols = ['text', 'label']
        self.train_df.drop([col for col in self.train_df.columns if \
                                col not in self.cols], axis=1, inplace=True)

        #print(self.train_df.columns)
        self.text = self.train_df.text
        self.label = self.train_df.label
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        #print('everything done')

    def __len__(self):
        return len(self.train_df.text)

    def __getitem__(self, idx):
        text = str(self.train_df.text[idx])
        text = " ".join(text.split())
        
        # inputs from tokenizer
        inputs = self.tokenizer.encode_plus(text,
                                            None,
                                            max_length=self.max_len,
                                            pad_to_max_length=True)

        print(f'\nType: {type(inputs)}')
        print(f'\ninput keys: {inputs.keys()}')

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        # conver all the above parameters to tensors and return as a dictionary
        return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.label[idx], dtype=torch.float) } 


#bd = BertDataset()
#print(bd.__getitem__(2))


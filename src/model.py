import os
import config
import transformers
import torch.nn as nn 
import dataset 

class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        
        #  Load the BERT model
        self.bert_model = transformers.BertModel.from_pretrained(config.BERT_PATH)
        #print(f'BERT model: {self.bert_model}')
        # Add a dropout layer to address overfitting
        self.bert_drop = nn.Dropout(0.25)
        # Add densel  layer, with one output neuron for binary classification
        self.out = nn.Linear(768, 1)

    def forward(self, ids,  attention_mask, token_type_ids):
        #  Take bert output of last hidden state from pooler  bert layer
        _, o2 = self.bert_model(ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

        # Add dropout on pooler layer output
        bo = self.bert_drop(o2)

        # Add final dense layer
        ouput = self.out(bo)

        return output


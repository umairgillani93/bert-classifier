import os 
import config
import torch
import flask
from flask import Flask 
from flask import request
from model import BERTModel
import torch.nn as nn 
import functools

app = Flask(__name__)

MODEL = None
DEVICE = config.DEVICE
PREDS = {}

def detect(text):
    '''
    implements the functionality and 
    pipeline for fake news detection model
    '''
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    text = str(text)
    text = " ".join(text.split)

    # perform tokenization
    inputs = tokenizer.encode_plust(
            text,
            None,
            add_special_tokens = True,
            max_length = max_len,
            truncation = True
            )

    # parse the tokenizer content
    ids = inputs['ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    # some code here...



    # convert all the tokenizer parameters to torch.tensor
    ids = torch.tensor(ids, dtype = torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype = torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype = torch.long).unsqueeze(0)

    # move all the parameters to device
    ids = ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, attention_mask=mask,
            token_type_ids = token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    
    return outputs[0][0]


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    text = request.args.get('text')
    start_time = time.time()
    true_news = detect(text)
    fake_news = 1 - ture_news # using log loss
    
    output = {}

    output['response'] = {
            "real": str(true_news),
            "fake": str(fake_news),
            "news": str(text),
            "time_taken": str(time.time() - start_time),
            }

    
if __name__ == '__main__':
    MODEL =  BERTModel()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(host='0.0.0.0', port=9999)


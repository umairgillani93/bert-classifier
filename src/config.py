import os
import transformers

# Created by Syed Umair Gillani; Umairgillani93@gmail.com
DEVICE = 'cpu'
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
BERT_PATH = os.getenv('HOME') + '/models/bert_model/model'
MODEL_PATH = os.getenv('HOME') + '/checkpoints/'
TRAINING_FILE = os.getenv('HOME') + '/datasets/fake-news/train.csv'
TOKENIZER = transformers.BertTokenizer.from_pretrained(os.getenv('HOME') + '/models/bert_model/tokenizer')
TRAIN_PATH = os.getenv('HOME') + '/datasets/fake_news/train.csv'
TEST_PATH = os.getenv('HOME') + '/datasets/fake_news/test.csv'


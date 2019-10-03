from __future__ import absolute_import, division, print_function, unicode_literals
import time
import datetime

# colab mode
# try:
#     %tensorflow_version 2.x
# except Exception:
#     pass
# !pip install tensorflow_probability==0.8.0rc0 --upgrade
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from utils import Mask, CustomSchedule, Trainer, translate
from data_loader import DataLoader
import datetime
from model import *
from tqdm import tqdm

# hyper paramaters
TRAIN_RATIO = 0.9
D_POINT_WISE_FF = 2048
D_MODEL = 512
ENCODER_COUNT = DECODER_COUNT = 6
EPOCHS = 20
ATTENTION_HEAD_COUNT = 8
DROPOUT_PROB = 0.1
BATCH_SIZE = 32
SEQ_MAX_LEN_SOURCE = 100
SEQ_MAX_LEN_TARGET = 100
BPE_VOCAB_SIZE = 32000


data_loader = DataLoader(
    dataset_name='wmt14/en-de',
    data_dir='./datasets'
)

source_data, target_data = data_loader.load_test(index=-1)
data = zip(source_data, target_data)

transformer = Transformer(
    input_vocab_size=BPE_VOCAB_SIZE,
    target_vocab_size=BPE_VOCAB_SIZE,
    encoder_count=ENCODER_COUNT,
    decoder_count=DECODER_COUNT,
    attention_head_count=ATTENTION_HEAD_COUNT,
    d_model=D_MODEL,
    d_point_wise_ff=D_POINT_WISE_FF,
    dropout_prob=DROPOUT_PROB
)

trainer = Trainer(model=transformer, checkpoint_dir='./checkpoints')
if trainer.checkpoint_manager.latest_checkpoint:
    print("Restored from {}".format(trainer.checkpoint_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

trainer.checkpoint.restore(
    trainer.checkpoint_manager.latest_checkpoint
)

translated_data = []

for source, target in tqdm(data):
    output = translate(source, data_loader, trainer, SEQ_MAX_LEN_TARGET)
    res = data_loader.sequences_to_texts([output.numpy().tolist()], mode='target')
    translated_data.append({
        'source': source,
        'target': target,
        'output': res
    })

import pickle
with open('translated_data.pickle', 'wb') as f:
    pickle.dump(pickle, f)

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import tensorflow as tf
from data_loader import DataLoader
from model import Transformer
from utils import CustomSchedule, Trainer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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

# for overfitting test hyper parameters
# BATCH_SIZE = 32
# EPOCHS = 100
DATA_LIMIT = None

GLOBAL_BATCH_SIZE = (BATCH_SIZE * 1)
print('GLOBAL_BATCH_SIZE ', GLOBAL_BATCH_SIZE)

data_loader = DataLoader(
    dataset_name='wmt14/en-de',
    data_dir='./datasets',
    batch_size=GLOBAL_BATCH_SIZE,
    bpe_vocab_size=BPE_VOCAB_SIZE,
    seq_max_len_source=SEQ_MAX_LEN_SOURCE,
    seq_max_len_target=SEQ_MAX_LEN_TARGET,
    data_limit=DATA_LIMIT,
    train_ratio=TRAIN_RATIO
)

dataset, val_dataset = data_loader.load()

transformer = Transformer(
    inputs_vocab_size=BPE_VOCAB_SIZE,
    target_vocab_size=BPE_VOCAB_SIZE,
    encoder_count=ENCODER_COUNT,
    decoder_count=DECODER_COUNT,
    attention_head_count=ATTENTION_HEAD_COUNT,
    d_model=D_MODEL,
    d_point_wise_ff=D_POINT_WISE_FF,
    dropout_prob=DROPOUT_PROB
)

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.losses.CategoricalCrossentropy(from_logits=True, reduction='none')

trainer = Trainer(
    model=transformer,
    dataset=dataset,
    loss_object=loss_object,
    optimizer=optimizer,
    batch_size=GLOBAL_BATCH_SIZE,
    vocab_size=BPE_VOCAB_SIZE,
    epoch=EPOCHS,
)

trainer.single_gpu_train()

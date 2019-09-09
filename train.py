from __future__ import absolute_import, division, print_function, unicode_literals

# colab mode
# try:
#     %tensorflow_version 2.x
# except Exception:
#     pass
# !pip install tensorflow_probability==0.8.0rc0 --upgrade

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import Mask, CustomSchedule, Trainer
from data_loader import DataLoader
from model import *

data_loader = DataLoader('wmt14/en-de', './datasets')

with tf.device('/CPU:0'):
    source_sequences, source_tokenizer, target_sequences, target_tokenizer = data_loader.load()
    
input_vocab_size = len(source_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1
print('input vocab size: ', input_vocab_size)
print('target vocab size: ', target_vocab_size)


# hyper paramaters
TRAIN_RATIO = 0.9
D_POINT_WISE_FF = 2048
D_MODEL = 512
ENCODER_COUNT = DECODER_COUNT = 6
EPOCHS = 20
ATTENTION_HEAD_COUNT = 8
DROPOUT_PROB = 0.1
BATCH_SIZE = 16

# for overfitting test hyper parameters
# BATCH_SIZE = 32
# EPOCHS = 100
DATA_LIMIT = None

with tf.device('/CPU:0'):
    source_sequences_train, source_sequences_val, target_sequences_train, target_sequences_val = train_test_split(
        source_sequences, target_sequences, train_size=TRAIN_RATIO
    )

    if DATA_LIMIT is not None:
        print('data size limit ON. limit size:', DATA_LIMIT)
        source_sequences_train = source_sequences_train[:DATA_LIMIT]
        target_sequences_train = target_sequences_train[:DATA_LIMIT]

    print('source_sequences_train', source_sequences_train.shape)
    print('source_sequences_val', source_sequences_val.shape)
    print('target_sequences_train', target_sequences_train.shape)
    print('target_sequences_val', target_sequences_val.shape)

    print('train set size: ', source_sequences_train.shape[0])
    print('validation set size: ', source_sequences_val.shape[0])
    TRAIN_SET_SIZE = source_sequences_train.shape[0]
    VALIDATION_SET_SIZE = source_sequences_val.shape[0]
    SEQUENCE_MAX_LENGTH = source_sequences_val.shape[1]
    
    
    
buffer_size = int(TRAIN_SET_SIZE * 0.3)
dataset = tf.data.Dataset.from_tensor_slices((source_sequences_train, target_sequences_train)).shuffle(buffer_size)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

transformer = Transformer(
     input_vocab_size=input_vocab_size,
     target_vocab_size=target_vocab_size,
     encoder_count=ENCODER_COUNT,
     decoder_count=DECODER_COUNT,
     attention_head_count=ATTENTION_HEAD_COUNT,
     d_model=D_MODEL,
     d_point_wise_ff=D_POINT_WISE_FF,
     dropout_prob=DROPOUT_PROB
)
 
learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

import time
import datetime

current_day = datetime.datetime.now().strftime("%Y%m%d")
train_log_dir = './logs/gradient_tape/' + current_day + '/train'
os.makedirs(train_log_dir, exist_ok=True)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

trainer = Trainer(
    model=transformer,
    loss_object=loss_object,
    optimizer=optimizer
)
if trainer.checkpoint_manager.latest_checkpoint:
    print("Restored from {}".format(trainer.checkpoint_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

trainer.checkpoint.restore(
    trainer.checkpoint_manager.latest_checkpoint
)

for epoch in range(EPOCHS):
    start = time.time()
    print('start learning')
    
    for (batch, (input, target)) in enumerate(dataset):
        loss = trainer.train_step(input, target)
        trainer.checkpoint.step.assign_add(1)
        if batch % 50 == 0:
            print("Epoch: {}, Batch: {}, Loss:{}, Accuracy: {}".format(epoch, batch, trainer.train_loss.result(), trainer.train_accuracy.result()))
    print("Epoch: {} Loss:{}, Accuracy: {}, time: {} sec". format(
        epoch, trainer.train_loss.result(), trainer.train_accuracy.result(), time.time() - start
    ))
    with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', trainer.train_loss.result(), step=epoch)
            tf.summary.scalar('train_accuracy', trainer.train_accuracy.result(), step=epoch)
    
    trainer.checkpoint_manager.save()
    
    trainer.train_loss.reset_states()
    trainer.train_accuracy.reset_states()
    trainer.validation_loss.reset_states()
    trainer.validation_accuracy.reset_states()
trainer.checkpoint_manager.save()

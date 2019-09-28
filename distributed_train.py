from __future__ import absolute_import, division, print_function, unicode_literals

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
from sklearn.model_selection import train_test_split
from utils import Mask, CustomSchedule, Trainer
from data_loader import DataLoader
import datetime
from model import *

data_loader = DataLoader('wmt14/en-de', './datasets')

with tf.device('/CPU:0'):
    source_sequences, target_sequences = data_loader.load()

# hyper paramaters
TRAIN_RATIO = 0.9
D_POINT_WISE_FF = 2048
D_MODEL = 512
ENCODER_COUNT = DECODER_COUNT = 6
EPOCHS = 20
ATTENTION_HEAD_COUNT = 8
DROPOUT_PROB = 0.1
BATCH_SIZE = 32

# for overfitting test hyper parameters
# BATCH_SIZE = 32
# EPOCHS = 100
DATA_LIMIT = None

with tf.device('/CPU:0'):
    source_sequences_train, source_sequences_val, target_sequences_train, target_sequences_val = train_test_split(
        source_texts, target_texts, train_size=TRAIN_RATIO
    )

    if DATA_LIMIT is not None:
        print('data size limit ON. limit size:', DATA_LIMIT)
        source_sequences_train = source_sequences_train[:DATA_LIMIT]
        target_sequences_train = target_sequences_train[:DATA_LIMIT]

    print('source_sequences_train', len(source_sequences_train))
    print('source_sequences_val', len(source_sequences_val))
    print('target_sequences_train', len(target_sequences_train))
    print('target_sequences_val', len(target_sequences_val))

    print('train set size: ', len(source_sequences_train))
    print('validation set size: ', len(source_sequences_val))
    TRAIN_SET_SIZE = len(source_sequences_train)
    VALIDATION_SET_SIZE = len(source_sequences_val)
    SEQUENCE_MAX_LENGTH = len(source_sequences_train[0])

strategy = tf.distribute.MirroredStrategy()

GLOBAL_BATCH_SIZE = (BATCH_SIZE *
                     strategy.num_replicas_in_sync)
print('GLOBAL_BATCH_SIZE ', GLOBAL_BATCH_SIZE)

buffer_size = int(TRAIN_SET_SIZE * 0.3)
dataset = tf.data.Dataset.from_tensor_slices((source_sequences_train, target_sequences_train)).shuffle(buffer_size)
dataset = dataset.batch(GLOBAL_BATCH_SIZE)
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
    optimizer=optimizer,
    batch_size=GLOBAL_BATCH_SIZE,
    distribute_strategy=strategy
)
if trainer.checkpoint_manager.latest_checkpoint:
    print("Restored from {}".format(trainer.checkpoint_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

trainer.checkpoint.restore(
    trainer.checkpoint_manager.latest_checkpoint
)


with strategy.scope():
    dataset = strategy.experimental_distribute_dataset(dataset)
    for epoch in range(EPOCHS):
        start = time.time()
        print('start learning')

        for (batch, (input, target)) in enumerate(dataset):
            loss = trainer.distributed_train_step(input, target)
            trainer.checkpoint.step.assign_add(1)
            if batch % 50 == 0:
                print("Epoch: {}, Batch: {}, Loss:{}, Accuracy: {}".format(epoch, batch, trainer.train_loss.result(),
                                                                           trainer.train_accuracy.result()))
            if batch % 10000 == 0 and batch != 0:
                trainer.checkpoint_manager.save()
        print("{} | Epoch: {} Loss:{}, Accuracy: {}, time: {} sec".format(
            datetime.datetime.now(), epoch, trainer.train_loss.result(), trainer.train_accuracy.result(), time.time() - start
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

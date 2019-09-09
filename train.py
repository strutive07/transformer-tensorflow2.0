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
from google.colab import files

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
            print(f"Epoch: {epoch}, Batch: {batch}, "
                  f"Loss:{trainer.train_loss.result()}, Accuracy: {trainer.train_accuracy.result()}")
    print(f"Epoch: {epoch} Loss:{trainer.train_loss.result()}, Accuracy: {trainer.train_accuracy.result()}, time: {time.time() - start} sec")
    with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', trainer.train_loss.result(), step=epoch)
            tf.summary.scalar('train_accuracy', trainer.train_accuracy.result(), step=epoch)
    
    trainer.checkpoint_manager.save()
    
    trainer.train_loss.reset_states()
    trainer.train_accuracy.reset_states()
    trainer.validation_loss.reset_states()
    trainer.validation_accuracy.reset_states()
trainer.checkpoint_manager.save()

class Trainer:
    def __init__(self, model, loss_object, optimizer, checkpoint_dir='./checkpoints'):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
#         self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)
        
        # metrics
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        self.validation_loss = tf.keras.metrics.Mean('validation_loss', dtype=tf.float32)
        self.validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('validation_accuracy')
        
    def train_step(self, input, target):
        target_include_start = target[:, :-1]
        target_include_end = target[:, 1:]
        encoder_padding_mask, look_ahead_mask, decoder_padding_mask = Mask.create_masks(
            input, target_include_start
        )
        
        with tf.GradientTape() as tape:
            pred = self.model.call(
                input=input,
                target=target_include_start,
                input_padding_mask=encoder_padding_mask,
                look_ahead_mask=look_ahead_mask,
                target_padding_mask=decoder_padding_mask,
                training=True
            )
            
            loss = self.loss_function(target_include_end, pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(target_include_end, pred)
        
        return tf.reduce_mean(loss)
    
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = self.loss_object(real, pred)
        
        mask = tf.cast(mask, dtype=loss.dtype)
        
        loss *= mask
        return tf.reduce_mean(loss)

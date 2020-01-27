from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

from data_loader import DataLoader
from model import Transformer
from utils import Trainer, calculate_bleu_score, translate

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

data_loader = DataLoader(
    dataset_name='wmt14/en-de',
    data_dir='./datasets'
)
data_loader.load_bpe_encoder()

source_data, target_data = data_loader.load_test(index=3)
_, target_data_path = data_loader.get_test_data_path(index=3)

data = zip(source_data, target_data)

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

trainer = Trainer(
    model=transformer,
    dataset=None,
    loss_object=None,
    optimizer=None,
    checkpoint_dir='./checkpoints'
)
if trainer.checkpoint_manager.latest_checkpoint:
    print("Restored from {}".format(trainer.checkpoint_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

trainer.checkpoint.restore(
    trainer.checkpoint_manager.latest_checkpoint
)


def do_translate(input_data):
    index = input_data[0]
    source = input_data[1][0]
    target = input_data[1][1]
    print(index)
    output = translate(source, data_loader, trainer, SEQ_MAX_LEN_TARGET)
    return {
        'source': source,
        'target': target,
        'output': output
    }


translated_data = []

for test_data in data:
    res = do_translate(test_data)
    translated_data.append(res['output'])

with open('translated_data', 'w') as f:
    f.write(str('\n'.join(translated_data)))

score, report = calculate_bleu_score(target_path='translated_data', ref_path=target_data_path)

import os
from urllib.request import urlretrieve

import sentencepiece
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DataLoader:
    DIR = None
    PATHS = {}
    BPE_VOCAB_SIZE = 0
    MODES = ['source', 'target']
    dictionary = {
        'source': {
            'token2idx': None,
            'idx2token': None,
        },
        'target': {
            'token2idx': None,
            'idx2token': None,
        }
    }
    CONFIG = {
        'wmt14/en-de': {
            'source_lang': 'en',
            'target_lang': 'de',
            'base_url': 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/',
            'train_files': ['train.en', 'train.de'],
            'vocab_files': ['vocab.50K.en', 'vocab.50K.de'],
            'dictionary_files': ['dict.en-de'],
            'test_files': [
                'newstest2012.en', 'newstest2012.de',
                'newstest2013.en', 'newstest2013.de',
                'newstest2014.en', 'newstest2014.de',
                'newstest2015.en', 'newstest2015.de',
            ]
        }
    }
    BPE_MODEL_SUFFIX = '.model'
    BPE_VOCAB_SUFFIX = '.vocab'
    BPE_RESULT_SUFFIX = '.sequences'
    SEQ_MAX_LEN = {
        'source': 100,
        'target': 100
    }
    DATA_LIMIT = None
    TRAIN_RATIO = 0.9
    BATCH_SIZE = 16

    source_sp = None
    target_sp = None

    def __init__(self, dataset_name, data_dir, batch_size=16, bpe_vocab_size=32000, seq_max_len_source=100,
                 seq_max_len_target=100, data_limit=None, train_ratio=0.9):
        if dataset_name is None or data_dir is None:
            raise ValueError('dataset_name and data_dir must be defined')
        self.DIR = data_dir
        self.DATASET = dataset_name
        self.BPE_VOCAB_SIZE = bpe_vocab_size
        self.SEQ_MAX_LEN['source'] = seq_max_len_source
        self.SEQ_MAX_LEN['target'] = seq_max_len_target
        self.DATA_LIMIT = data_limit
        self.TRAIN_RATIO = train_ratio
        self.BATCH_SIZE = batch_size

        self.PATHS['source_data'] = os.path.join(self.DIR, self.CONFIG[self.DATASET]['train_files'][0])
        self.PATHS['source_bpe_prefix'] = self.PATHS['source_data'] + '.segmented'

        self.PATHS['target_data'] = os.path.join(self.DIR, self.CONFIG[self.DATASET]['train_files'][1])
        self.PATHS['target_bpe_prefix'] = self.PATHS['target_data'] + '.segmented'

    def load(self, custom_dataset=False):
        if custom_dataset:
            print('#1 download data')
            self.download_dataset()
        else:
            print('#1 use custom dataset.')

        print('#2 parse data')
        source_data = self.parse_data_and_save(self.PATHS['source_data'])
        target_data = self.parse_data_and_save(self.PATHS['target_data'])

        print('#3 train bpe')

        self.train_bpe(self.PATHS['source_data'], self.PATHS['source_bpe_prefix'])
        self.train_bpe(self.PATHS['target_data'], self.PATHS['target_bpe_prefix'])

        print('#4 load bpe vocab')

        self.dictionary['source']['token2idx'], self.dictionary['source']['idx2token'] = self.load_bpe_vocab(
            self.PATHS['source_bpe_prefix'] + self.BPE_VOCAB_SUFFIX)
        self.dictionary['target']['token2idx'], self.dictionary['target']['idx2token'] = self.load_bpe_vocab(
            self.PATHS['target_bpe_prefix'] + self.BPE_VOCAB_SUFFIX)

        print('#5 encode data with bpe')
        source_sequences = self.texts_to_sequences(
            self.sentence_piece(
                source_data,
                self.PATHS['source_bpe_prefix'] + self.BPE_MODEL_SUFFIX,
                self.PATHS['source_bpe_prefix'] + self.BPE_RESULT_SUFFIX
            ),
            mode="source"
        )
        target_sequences = self.texts_to_sequences(
            self.sentence_piece(
                target_data,
                self.PATHS['target_bpe_prefix'] + self.BPE_MODEL_SUFFIX,
                self.PATHS['target_bpe_prefix'] + self.BPE_RESULT_SUFFIX
            ),
            mode="target"
        )

        print('source sequence example:', source_sequences[0])
        print('target sequence example:', target_sequences[0])

        if self.TRAIN_RATIO == 1.0:
            source_sequences_train = source_sequences
            source_sequences_val = []
            target_sequences_train = target_sequences
            target_sequences_val = []
        else:
            source_sequences_train, source_sequences_val, target_sequences_train, target_sequences_val = train_test_split(
                source_sequences, target_sequences, train_size=self.TRAIN_RATIO
            )

        if self.DATA_LIMIT is not None:
            print('data size limit ON. limit size:', self.DATA_LIMIT)
            source_sequences_train = source_sequences_train[:self.DATA_LIMIT]
            target_sequences_train = target_sequences_train[:self.DATA_LIMIT]

        print('source_sequences_train', len(source_sequences_train))
        print('source_sequences_val', len(source_sequences_val))
        print('target_sequences_train', len(target_sequences_train))
        print('target_sequences_val', len(target_sequences_val))

        print('train set size: ', len(source_sequences_train))
        print('validation set size: ', len(source_sequences_val))

        train_dataset = self.create_dataset(
            source_sequences_train,
            target_sequences_train
        )
        if self.TRAIN_RATIO == 1.0:
            val_dataset = None
        else:
            val_dataset = self.create_dataset(
                source_sequences_val,
                target_sequences_val
            )

        return train_dataset, val_dataset

    def load_test(self, index=0, custom_dataset=False):
        if index < 0 or index >= len(self.CONFIG[self.DATASET]['test_files']) // 2:
            raise ValueError('test file index out of range. min: 0, max: {}'.format(len(self.CONFIG[self.DATASET]['test_files']) // 2 - 1))
        if custom_dataset:
            print('#1 download data')
            self.download_dataset()
        else:
            print('#1 use custom dataset.')

        print('#2 parse data')

        source_test_data_path, target_test_data_path = self.get_test_data_path(index)

        source_data = self.parse_data_and_save(source_test_data_path)
        target_data = self.parse_data_and_save(target_test_data_path)

        print('#3 load bpe vocab')

        self.dictionary['source']['token2idx'], self.dictionary['source']['idx2token'] = self.load_bpe_vocab(
            self.PATHS['source_bpe_prefix'] + self.BPE_VOCAB_SUFFIX)
        self.dictionary['target']['token2idx'], self.dictionary['target']['idx2token'] = self.load_bpe_vocab(
            self.PATHS['target_bpe_prefix'] + self.BPE_VOCAB_SUFFIX)

        return source_data, target_data

    def get_test_data_path(self, index):
        source_test_data_path = os.path.join(self.DIR, self.CONFIG[self.DATASET]['test_files'][index * 2])
        target_test_data_path = os.path.join(self.DIR, self.CONFIG[self.DATASET]['test_files'][index * 2 + 1])
        return source_test_data_path, target_test_data_path

    def download_dataset(self):
        for file in (self.CONFIG[self.DATASET]['train_files']
                     + self.CONFIG[self.DATASET]['vocab_files']
                     + self.CONFIG[self.DATASET]['dictionary_files']
                     + self.CONFIG[self.DATASET]['test_files']):
            self._download("{}{}".format(self.CONFIG[self.DATASET]['base_url'], file))

    def _download(self, url):
        path = os.path.join(self.DIR, url.split('/')[-1])
        if not os.path.exists(path):
            with TqdmCustom(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=url) as t:
                urlretrieve(url, path, t.update_to)

    def parse_data_and_save(self, path):
        print('load data from {}'.format(path))
        with open(path, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')

        if lines is None:
            raise ValueError('Vocab file is invalid')

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return lines

    def train_bpe(self, data_path, model_prefix):
        model_path = model_prefix + self.BPE_MODEL_SUFFIX
        vocab_path = model_prefix + self.BPE_VOCAB_SUFFIX

        if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
            print('bpe model does not exist. train bpe. model path:', model_path, ' vocab path:', vocab_path)
            train_source_params = "--input={} \
                --pad_id=0 \
                --unk_id=1 \
                --bos_id=2 \
                --eos_id=3 \
                --model_prefix={} \
                --vocab_size={} \
                --model_type=bpe ".format(
                data_path,
                model_prefix,
                self.BPE_VOCAB_SIZE
            )
            sentencepiece.SentencePieceTrainer.Train(train_source_params)
        else:
            print('bpe model exist. load bpe. model path:', model_path, ' vocab path:', vocab_path)
            
    def load_bpe_encoder(self):
        self.dictionary['source']['token2idx'], self.dictionary['source']['idx2token'] =  self.load_bpe_vocab(self.PATHS['source_bpe_prefix'] + self.BPE_VOCAB_SUFFIX)
        self.dictionary['target']['token2idx'], self.dictionary['target']['idx2token'] =  self.load_bpe_vocab(self.PATHS['target_bpe_prefix'] + self.BPE_VOCAB_SUFFIX)

    def sentence_piece(self, source_data, source_bpe_model_path, result_data_path):
        sp = sentencepiece.SentencePieceProcessor()
        sp.load(source_bpe_model_path)

        if os.path.exists(result_data_path):
            print('encoded data exist. load data. path:', result_data_path)
            with open(result_data_path, 'r', encoding='utf-8') as f:
                sequences = f.read().strip().split('\n')
                return sequences

        print('encoded data does not exist. encode data. path:', result_data_path)
        sequences = []
        with open(result_data_path, 'w') as f:
            for sentence in tqdm(source_data):
                pieces = sp.EncodeAsPieces(sentence)
                sequence = " ".join(pieces)
                sequences.append(sequence)
                f.write(sequence + "\n")
        return sequences

    def encode_data(self, input, mode='source'):
        if mode not in self.MODES:
            ValueError('not allowed mode.')

        if mode == 'source':
            if self.source_sp is None:
                self.source_sp = sentencepiece.SentencePieceProcessor()
                self.source_sp.load(self.PATHS['source_bpe_prefix'] + self.BPE_MODEL_SUFFIX)

            pieces = self.source_sp.EncodeAsPieces(input)
            sequence = " ".join(pieces)

        elif mode == 'target':
            if self.target_sp is None:
                self.target_sp = sentencepiece.SentencePieceProcessor()
                self.target_sp.load(self.PATHS['target_bpe_prefix'] + self.BPE_MODEL_SUFFIX)

            pieces = self.target_sp.EncodeAsPieces(input)
            sequence = " ".join(pieces)

        else:
            ValueError('not allowed mode.')

        return sequence

    def load_bpe_vocab(self, bpe_vocab_path):
        with open(bpe_vocab_path, 'r') as f:
            vocab = [line.split()[0] for line in f.read().splitlines()]

        token2idx = {}
        idx2token = {}

        for idx, token in enumerate(vocab):
            token2idx[token] = idx
            idx2token[idx] = token
        return token2idx, idx2token

    def texts_to_sequences(self, texts, mode='source'):
        if mode not in self.MODES:
            ValueError('not allowed mode.')

        sequences = []
        for text in texts:
            text_list = ["<s>"] + text.split() + ["</s>"]

            sequence = [
                self.dictionary[mode]['token2idx'].get(
                    token, self.dictionary[mode]['token2idx']["<unk>"]
                )
                for token in text_list
            ]
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences, mode='source'):
        if mode not in self.MODES:
            ValueError('not allowed mode.')

        texts = []
        for sequence in sequences:
            if mode == 'source':
                if self.source_sp is None:
                    self.source_sp = sentencepiece.SentencePieceProcessor()
                    self.source_sp.load(self.PATHS['source_bpe_prefix'] + self.BPE_MODEL_SUFFIX)
                text = self.source_sp.DecodeIds(sequence)
            else:
                if self.target_sp is None:
                    self.target_sp = sentencepiece.SentencePieceProcessor()
                    self.target_sp.load(self.PATHS['target_bpe_prefix'] + self.BPE_MODEL_SUFFIX)
                text = self.target_sp.DecodeIds(sequence)
            texts.append(text)
        return texts

    def create_dataset(self, source_sequences, target_sequences):
        new_source_sequences = []
        new_target_sequences = []
        for source, target in zip(source_sequences, target_sequences):
            if len(source) > self.SEQ_MAX_LEN['source']:
                continue
            if len(target) > self.SEQ_MAX_LEN['target']:
                continue
            new_source_sequences.append(source)
            new_target_sequences.append(target)

        source_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences=new_source_sequences, maxlen=self.SEQ_MAX_LEN['source'], padding='post'
        )
        target_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences=new_target_sequences, maxlen=self.SEQ_MAX_LEN['target'], padding='post'
        )
        buffer_size = int(source_sequences.shape[0] * 0.3)
        dataset = tf.data.Dataset.from_tensor_slices(
            (source_sequences, target_sequences)
        ).shuffle(buffer_size)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


class TqdmCustom(tqdm):

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

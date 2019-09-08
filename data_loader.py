import os
from urllib.request import urlretrieve

from tqdm import tqdm

import tensorflow as tf
import pickle

class DataLoader:
    DIR = None
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

    def __init__(self, dataset_name, data_dir):
        if dataset_name is None or data_dir is None:
            raise ValueError('dataset_name and data_dir must be defined')
        self.DIR = data_dir
        self.DATASET = dataset_name

    def load(self):
        pickle_data_path = os.path.join(self.DIR, 'data.pickle')
        if os.path.exists(pickle_data_path):
            with open(pickle_data_path, 'rb') as f:
                data_dict = pickle.load(f)
                return (
                    data_dict['source_sequences'],
                    data_dict['source_tokenizer'],
                    data_dict['target_sequences'],
                    data_dict['target_tokenizer']
                )
        else:
            print('#1 download data')
            self.download_dataset()

            print('#2 load data')
            word2idx_source, idx2word_source, word2idx_target, idx2word_target = self.load_vocab()

            source_data = self.load_data(os.path.join(self.DIR, self.CONFIG[self.DATASET]['train_files'][0]))
            target_data = self.load_data(os.path.join(self.DIR, self.CONFIG[self.DATASET]['train_files'][1]))

            print('#3 tokenize data')
            source_sequences, source_tokenizer = self.tokenize(source_data, word2idx_source, idx2word_source)
            target_sequences, target_tokenizer = self.tokenize(target_data, word2idx_target, idx2word_target)

            return source_sequences, source_tokenizer, target_sequences, target_tokenizer

    def download_dataset(self):
        for file in (self.CONFIG[self.DATASET]['train_files']
                     + self.CONFIG[self.DATASET]['vocab_files']
                     + self.CONFIG[self.DATASET]['dictionary_files']
                     + self.CONFIG[self.DATASET]['test_files']):
            self._download(f"{self.CONFIG[self.DATASET]['base_url']}{file}")

    def _download(self, url):
        path = os.path.join(self.DIR, url.split('/')[-1])
        if not os.path.exists(path):
            with TqdmCustom(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=url) as t:
                urlretrieve(url, path, t.update_to)

    def load_vocab(self):
        word2idxs = {
            self.CONFIG[self.DATASET]['source_lang']: {},
            self.CONFIG[self.DATASET]['target_lang']: {}
        }

        idx2words = {
            self.CONFIG[self.DATASET]['source_lang']: {},
            self.CONFIG[self.DATASET]['target_lang']: {}
        }

        for vocab_file in self.CONFIG[self.DATASET]['vocab_files']:
            word2idx = {}
            idx2word = {}

            lang = vocab_file[-2:]
            path = os.path.join(self.DIR, vocab_file)

            with open(path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')

            if lines is None:
                raise ValueError('Vocab file is invalid')

            # set padding to index 0
            word2idx['<pad>'] = 0
            idx2word[0] = '<pad>'

            for index, word in enumerate(lines, start=1):
                word2idx[word] = index
                idx2word[index] = word

            word2idxs[lang] = word2idx
            idx2words[lang] = idx2word

        return (word2idxs[self.CONFIG[self.DATASET]['source_lang']],
                idx2words[self.CONFIG[self.DATASET]['source_lang']],
                word2idxs[self.CONFIG[self.DATASET]['target_lang']],
                idx2words[self.CONFIG[self.DATASET]['target_lang']])

    def load_data(self, path):
        print(f'load data from {path}')
        with open(path, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')

        if lines is None:
            raise ValueError('Vocab file is invalid')

        return [f"<s> {line} </s>" for line in tqdm(lines)]

    def tokenize(self, data, word2idx, idx2word):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
        tokenizer.word_index = word2idx
        tokenizer.index_word = idx2word
        sequences = tokenizer.texts_to_sequences(data)

        max_length = max(len(sequence) for sequence in tqdm(sequences))

        sequences = tf.keras.preprocessing.sequence.pad_sequences( # len(sequences), max_length
            sequences=sequences, maxlen=max_length, padding='post'
        )
        return sequences, tokenizer
    # TODO train, validation split function


class TqdmCustom(tqdm):

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

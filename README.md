# Transformer-tensorflow2.0

[attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) (transformer) in tensorflow 2.0

[paper review(pdf)](https://github.com/strutive07/transformer-tensorflow2.0/blob/master/Attention%20is%20all%20you%20need.pdf)

[colab guide](https://colab.research.google.com/github/strutive07/transformer-tensorflow2.0/blob/master/transformer_implement_tf2_0.ipynb)

[Download pre-trained model(checkpoint)](https://drive.google.com/file/d/1jsY7WMI9EU5ifhcxV_sMpK8znPA1mvkf/view?usp=sharing)

[Download pre-trained bpe data](https://drive.google.com/drive/folders/1YUABrVUz3oGKgGfMJNWQl0WCP_nVjhiS?usp=sharing)

[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/strutive07/transformer-tensorflow2.0/?ref=repository-badge)

## How to train

1. Install enviornments

    bash ubuntu16_04_cuda10_cudnn7_tensorflow2.0_install.sh
    
2. Training

- Single GPU training
    1. Change hyper parameter in train.py
    2. Run training script

    ```bash
    python train.py
    ```

    

- Multi GPU training
    1. Change hyper parameter in distributed_train.py
    2. Run training script

    ```bash
    python distributed_train.py
    ```

3. Test
- if you did not train bpe, train bpe model or download pre-trained bpe model. LINK: [Download pre-trained bpe data](https://drive.google.com/drive/folders/1YUABrVUz3oGKgGfMJNWQl0WCP_nVjhiS?usp=sharing). You should save it in *top dataset directory*.
example: ./dataset/train.en.segmented.vocab and so on.


## How to add dataset

Add data config to `data_loader.py`

```python
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
```

If you want to add custom dataset, add data config like below and add `custom_dataset` parameter to DataLoader.load

```python
CONFIG = {
        'wmt14/en-de': {
            'source_lang': 'en',
            'target_lang': 'de',
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

dataset, val_dataset = data_loader.load(custom_dataset=True)
```



## BLEU Score

| Test Dataset | BLEU Score |
| ------------ | ---------- |
| newstest2013 | 23.3       |
| newstest2014 | 22.85      |
| newstest2015 | 25.33      |

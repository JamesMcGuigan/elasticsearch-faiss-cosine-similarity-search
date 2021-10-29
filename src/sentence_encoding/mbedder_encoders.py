#!/usr/bin/env python3
import time
from typing import Collection, List

import numpy as np
import torch
from Mbedder.embeddings import (BertEmbedding, DistilBertEmbedding, XLMEmbedding, XLMRobertaEmbedding)
from transformers import BertTokenizer, DistilBertTokenizer, XLMRobertaTokenizer, XLMTokenizer

from src.sentence_encoding.sentence_encoder import SentenceEncoder


class MbedderSentenceEncoder(SentenceEncoder):
    """
    Class for encoding sentences as sentence embeddings using Mbedder
    - https://github.com/monkeysforever/Mbedder

    Mbedder models:
    - https://huggingface.co/transformers/pretrained_models.html
    - https://huggingface.co/models
    - venv/lib/python3.8/site-packages/Mbedder/embeddings.py

    Code examples:
    - https://github.com/monkeysforever/Mbedder/blob/master/tests/test_embeddings.py
    """
    def __init__(self, model_id: str=''):
        super().__init__()
        self.model_id  = model_id
        self.tokenizer = None
        self.embedding = None


    def encode_sentences(
            self,
            sentences: List[str],
            language: str = None
    ) -> np.ndarray:
        """
        Encodes a list of sentences using Google's LaBSE model. Each
        sentence is transformed into a float vector (embedding).

        Args:
            sentences (List[str]): List of sentences to encode.
            language (str, optional): This parameter is ignored

        Returns:
            Collection[Any]: 2D NumPy array whose rows represent the
              encoding for a particular sentence and whose columns
              represent the different dimensions of the encoding.
        """


        # Sentences may be of different lengths, thus must be encoded individually
        # This may potentially be much slower than using batch mode
        embeddings = []
        input_embedding = self.tokenizer(sentences)
        for n in range(len(input_embedding['input_ids'])):
            input_ids = torch.tensor(input_embedding['input_ids'][n]).unsqueeze(0)
            mask      = torch.tensor(input_embedding['attention_mask'][n]).unsqueeze(0)
            sentence_embedding, token_embedding = self.embedding(input=input_ids, mask=mask)
            embeddings.append( sentence_embedding )
        embeddings = torch.cat(embeddings).numpy()
        return embeddings


class BertSentenceEncoder(MbedderSentenceEncoder):
    """
    'bert-base-multilingual-cased'
    (New, recommended) 12-layer, 768-hidden, 12-heads, 179M parameters.
    Trained on cased text in the top 104 languages with the largest Wikipedias
    - https://huggingface.co/transformers/pretrained_models.html
    - Details: https://github.com/google-research/bert/blob/master/multilingual.md
    """
    def __init__(self, model_id: str = 'bert-base-multilingual-cased'):
        super().__init__(model_id)
        self.tokenizer = BertTokenizer.from_pretrained(model_id)
        self.embedding = BertEmbedding.from_pretrained(model_id)


class DistilBERTSentenceEncoder(MbedderSentenceEncoder):
    """
    distilbert-base-multilingual-cased
    6-layer, 768-hidden, 12-heads, 134M parameters
    The multilingual DistilBERT model distilled from the Multilingual BERT model bert-base-multilingual-cased checkpoint.
    - https://huggingface.co/transformers/pretrained_models.html
    """
    def __init__(self, model_id: str = 'distilbert-base-multilingual-cased'):
        super().__init__(model_id)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_id)
        self.embedding = DistilBertEmbedding.from_pretrained(model_id)



class XLMSentenceEncoder(MbedderSentenceEncoder):
    """
    xlm-mlm-100-1280
    16-layer, 1280-hidden, 16-heads
    XLM model trained with MLM (Masked Language Modeling) on 100 languages.
    - https://huggingface.co/transformers/pretrained_models.html
    """
    def __init__(self, model_id: str = 'xlm-mlm-100-1280'):
        super().__init__(model_id)
        self.tokenizer = XLMTokenizer.from_pretrained(model_id)
        self.embedding = XLMEmbedding.from_pretrained(model_id)



class XLMRoBERTaSentenceEncoder(MbedderSentenceEncoder):
    """
    xlm-roberta-large
    ~550M parameters with 24-layers, 1024-hidden-state, 4096 feed-forward hidden-state, 16-heads,
    Trained on 2.5 TB of newly created clean CommonCrawl data in 100 languages
    - https://huggingface.co/transformers/pretrained_models.html

    XLMRobertaTokenizer requires the SentencePiece library
        - pip install sentencepiece
        - https://github.com/google/sentencepiece#installation
    """
    def __init__(self, model_id: str = 'xlm-roberta-large'):
        super().__init__(model_id)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_id)
        self.embedding = XLMRobertaEmbedding.from_pretrained(model_id)



### TODO: MBartEmbedding not defined in Mbedder
# class MBertSentenceEncoder(MbedderSentenceEncoder):
#     """
#     facebook/mbart-large-cc25
#     24-layer, 1024-hidden, 16-heads, 610M parameters
#     mBART (bart-large architecture) model trained on 25 languages’ monolingual corpus
#     - https://huggingface.co/transformers/pretrained_models.html
#     """
#     def __init__(self, model_id: str = 'facebook/mbart-large-cc25'):
#         super().__init__(model_id)
#         self.tokenizer = MBartTokenizer.from_pretrained(model_id)
#         self.embedding = MBartEmbedding.from_pretrained(model_id)



# BertSentenceEncoder        = (32, 768)  in  6.1s load  1.9s embed
# DistilBERTSentenceEncoder  = (32, 768)  in  5.5s load  0.8s embed
# XLMSentenceEncoder         = (32, 1280) in 14.4s load  4.9s embed
# XLMRoBERTaSentenceEncoder  = (32, 1024) in 15.6s load  6.5s embed
if __name__ == '__main__':
    from config import config
    from src.utils.files import file_readlines

    # sentences = [
    #     "Mary had a little lamb",
    #     "Cat in the hat",
    #     "Pen Pineapple Apple Pen",
    # ]
    sentences = file_readlines(config['datasets']['sentences']['en']) + file_readlines(config['datasets']['sentences']['it'])
    encoders = [
        BertSentenceEncoder,
        DistilBERTSentenceEncoder,
        XLMSentenceEncoder,
        XLMRoBERTaSentenceEncoder,
    ]
    for encoder_class in encoders:
        time_start, time_end = {}, {}

        time_start['load']  = time.perf_counter()
        encoder = encoder_class()  # ignore load times
        time_end['load']    = time.perf_counter()

        time_start['embed'] = time.perf_counter()
        embedding = encoder.encode_sentences(sentences)
        time_end['embed']   = time.perf_counter()

        time_taken = { key: round(time_end[key] - time_start[key], 1) for key in time_end.keys() }
        print(f"{encoder_class.__name__:26s} = {str(embedding.shape):10s} in {time_taken['load']:4.1f}s load {time_taken['embed']:4.1f}s embed")

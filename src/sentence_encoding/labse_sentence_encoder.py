# Code adapted from: https://tfhub.dev/google/LaBSE/1

from typing import Any, Collection, List, Tuple

import bert
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization.bert_tokenization import FullTokenizer
from fastcache import clru_cache

from config import config
from src.sentence_encoding.sentence_encoder import SentenceEncoder

# BUGFIX: Prevent tensorflow from allocating all GPU memory
# https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
for gpu in tf.config.list_physical_devices('GPU') or []:
    try:    tf.config.experimental.set_memory_growth(gpu, True)
    except: pass


class LabseSentenceEncoder(SentenceEncoder):
    """Class for encoding sentences as sentence embeddings using 
    Google's LaBSE model (https://arxiv.org/abs/2007.01852).

    Args:
        model_url (str): String to load a saved model using hub.load().
          Can be a file path or a URL.
        max_seq_length (int, Optional): Maximum length of input 
          sequences. Shorter sequences will be padded, whereas longer 
          sequences will be truncated. Defaults to 64.
    """

    def __init__(self,
                 model_url:      str = config['labse']['tensorflow_model_dir'],
                 max_seq_length: int = config['labse']['max_seq_length']
    ):
        super().__init__()
        self.model_url      = model_url
        self.max_seq_length = max_seq_length
        self.labse_model, self.bert_tokenizer = self._get_labse_model(self.model_url, self.max_seq_length)


    # NOTE: Needs to to cached as @staticmethod else throws exception when called by multiple instances
    @staticmethod
    @clru_cache(None)
    def _get_labse_model(model_url, max_seq_length) -> Tuple[tf.keras.Model, FullTokenizer]:
        """Builds and returns Google's LaBSE model.

        Returns:
            Tuple[tf.keras.Model, FullTokenizer]: Pair containing the 
              LaBSE model and a BERT tokenizer.
        """

        # Define inputs
        input_word_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids"
        )
        input_mask = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name="input_mask"
        )
        segment_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name="segment_ids"
        )

        # LaBSE layer
        labse_layer = hub.KerasLayer(model_url, trainable=False)
        pooled_output, _ = labse_layer([
            input_word_ids, input_mask, segment_ids
        ])

        # Define model
        labse_model = tf.keras.Model(
            inputs=[input_word_ids, input_mask, segment_ids], 
            outputs=pooled_output
        )

        # Define BERT tokenizer
        vocab_file = labse_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case  = labse_layer.resolved_object.do_lower_case.numpy()
        bert_tokenizer = bert.bert_tokenization.FullTokenizer(
            vocab_file, do_lower_case
        )
        
        return (labse_model, bert_tokenizer)


    def _labse_cast_input(
        self,
        input_strings: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts "raw" inputs (list of sentences) in a format which 
        is suitable for the LaBSE model.

        Args:
            input_strings (List[str]): List of sentences to feed to the 
              LaBSE model.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Triplet 
              containing (possibly padded) input ids, input masks (to 
              distinguish between "real" tokens and padding tokens), and
              finally segment ids (LaBSE is based on BERT, so we have to
              adhere to BERT's expected input format).
        """

        input_ids_all, input_mask_all, segment_ids_all = [], [], []
        for input_string in input_strings:
            # Tokenize input
            input_tokens = (['[CLS]'] 
                            + self.bert_tokenizer.tokenize(input_string) 
                            + ['[SEP]'])
            input_ids = self.bert_tokenizer.convert_tokens_to_ids(input_tokens)
            sequence_length = min(len(input_ids), self.max_seq_length)

            # Padding or truncation
            if len(input_ids) >= self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]
            else:
                input_ids = (input_ids 
                             + [0] * (self.max_seq_length - len(input_ids)))

            # Masking
            input_mask = ([1] * sequence_length
                          + [0] * (self.max_seq_length - sequence_length))

            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            segment_ids_all.append([0] * self.max_seq_length)

        return (
            tf.convert_to_tensor(input_ids_all,   dtype=tf.int32),
            tf.convert_to_tensor(input_mask_all,  dtype=tf.int32),
            tf.convert_to_tensor(segment_ids_all, dtype=tf.int32),
        )


    # WARNING: tensorflow: 7 out of the last 14 calls to <function recreate_function.<locals>.restored_function_body>
    # triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to
    # (1) creating @tf.function repeatedly in a loop,
    # (2) passing tensors with different shapes,
    # (3) passing Python objects instead of tensors.
    # For (1), please define your @tf.function outside of the loop.
    # For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing.
    # For (3), please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.
    # NOTE: Root issue is that batch_size differs by document
    def encode_sentences(
        self, 
        sentences: List[str], 
        language: str = None  # pylint: disable=unused-argument
    ) -> Collection[Any]:
        """Encodes a list of sentences using Google's LaBSE model. Each
        sentence is transformed into a float vector (embedding).

        Args:
            sentences (List[str]): List of sentences to encode.
            language (str, optional): This parameter is ignored, as 
              LaBSE doesn't need language information to encode 
              sentences. Defaults to None.

        Returns:
            Collection[Any]: 2D NumPy array whose rows represent the 
              encoding for a particular sentence and whose columns 
              represent the different dimensions of the encoding.
        """

        # Prepare input
        input_ids, input_mask, segment_ids = self._labse_cast_input(sentences)
        # Encode the sentences
        encoding = self.labse_model([ input_ids, input_mask, segment_ids ])
        
        return encoding

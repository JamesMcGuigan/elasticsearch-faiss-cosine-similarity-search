from typing import List, Tuple, Union

import bert
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization.bert_tokenization import FullTokenizer
from fastcache import clru_cache

from config import config
from src.utils.punkt_tokenizer import punkt_tokenize_sentences

# Code adapted from: https://tfhub.dev/google/LaBSE/1

# Question: 64 was example code default - is this the optimal number???
# Hardcoded value as this needs to be set the same for get_labse_model() 
# and labse_cast_input()
max_seq_length = 64

# BUG:  CUDA GPU memory is exceeded if both laser and labse are loaded 
# together
@clru_cache(None)
def get_labse_model(
    model_url: str = None
) -> Tuple[tf.keras.Model, tf.keras.layers.Layer, FullTokenizer]:
    # TODO: Docstring

    model_url = model_url or config['labse']['tensorflow_model_dir']

    # Define input
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
    labse_layer      = hub.KerasLayer(model_url, trainable=True)
    pooled_output, _ = labse_layer([input_word_ids, input_mask, segment_ids])
    
    # The embedding is l2 normalized
    pooled_output = tf.keras.layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1)
    )(pooled_output)
    
    # Define model
    labse_model = tf.keras.Model(
        inputs=[input_word_ids, input_mask, segment_ids], outputs=pooled_output
    )
    
    # Define BERT tokenizer
    vocab_file     = labse_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case  = labse_layer.resolved_object.do_lower_case.numpy()
    bert_tokenizer = bert.bert_tokenization.FullTokenizer(
        vocab_file, do_lower_case
    )
    
    return (labse_model, labse_layer, bert_tokenizer)


def labse_cast_input(
    input_strings: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: Docstring

    _, _, bert_tokenizer = get_labse_model()

    input_ids_all, input_mask_all, segment_ids_all = [], [], []
    for input_string in input_strings:
        # Tokenize input
        input_tokens = (["[CLS]"] 
                        + bert_tokenizer.tokenize(input_string) 
                        + ["[SEP]"])
        input_ids       = bert_tokenizer.convert_tokens_to_ids(input_tokens)
        sequence_length = min(len(input_ids), max_seq_length)

        # Padding or truncation
        if len(input_ids) >= max_seq_length:
            input_ids = input_ids[:max_seq_length]
        else:
            input_ids = input_ids + [0] * (max_seq_length - len(input_ids))

        input_mask = ( [1] * sequence_length
                     + [0] * (max_seq_length - sequence_length))

        input_ids_all.append(input_ids)
        input_mask_all.append(input_mask)
        segment_ids_all.append([0] * max_seq_length)

    return (
        tf.convert_to_tensor(input_ids_all,   dtype=tf.int32),
        tf.convert_to_tensor(input_mask_all,  dtype=tf.int32),
        tf.convert_to_tensor(segment_ids_all, dtype=tf.int32),
    )


### This is the main export function
# WARNING: tensorflow: 7 out of the last 14 calls to <function recreate_function.<locals>.restored_function_body>
# triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to
# (1) creating @tf.function repeatedly in a loop,
# (2) passing tensors with different shapes,
# (3) passing Python objects instead of tensors.
# For (1), please define your @tf.function outside of the loop.
# For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing.
# For (3), please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.
# NOTE: Root issue is that batch_size differs by document
def labse_encode(text: Union[str, List[str]]) -> np.ndarray:
    # TODO: Docstring

    if isinstance(text, str):
        sentences = punkt_tokenize_sentences(text)
    else:
        sentences = text

    input_ids, input_mask, segment_ids = labse_cast_input(sentences)
    # print(f'input_ids, input_mask, segment_ids: {input_ids.shape} '
    #       f'{input_mask.shape} {segment_ids.shape}')

    labse_model, _, _ = get_labse_model()
    encoding = labse_model([ input_ids, input_mask, segment_ids ])
    return encoding

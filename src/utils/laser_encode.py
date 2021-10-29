# NOTE: Change this to the directory where you downloaded the pre-trained model
from pathlib import Path
from typing import List, Union

import numpy as np
from fastcache import clru_cache
from laserembeddings import Laser

from config import config
from src.utils.fasttest_model import language_detect
from src.utils.punkt_tokenizer import punkt_tokenize_sentences

base_dir          = Path(config['laser']['base_dir'])
path_to_bpe_codes = Path(config['laser']['bpe_codes'])
path_to_bpe_vocab = Path(config['laser']['bpe_vocab'])
path_to_encoder   = Path(config['laser']['encoder'])

# Instantiate encoder
# BUG: CUDA GPU memory is exceeded if both laser and labse are loaded 
# together
@clru_cache(None)
def get_laser_model():
    laser_model = Laser(
        bpe_codes = config['laser']['bpe_codes'],
        bpe_vocab = config['laser']['bpe_vocab'],
        encoder   = config['laser']['encoder'],
        tokenizer_options = None,
        embedding_options = None
    )
    return laser_model

def laser_encode(text: Union[str, List[str]], lang=None) -> np.ndarray:
    """
    Encodes a corpus of text using LASER
    :param text: Large block of text (will be tokenized), or list of pre-tokenized sentences
    :param lang: 2 digit language code (optional autodetect)
    :return:     embedding matrix
    """
    laser_model = get_laser_model()
    lang = lang or language_detect(text, threshold=0.0)
    
    if isinstance(text, str):
        sentences = punkt_tokenize_sentences(text, lang=lang)
    else:
        sentences = text

    embedding = laser_model.embed_sentences(sentences, lang=lang)
    return embedding

# DOCS: https://fasttext.cc/docs/en/language-identification.html
# DOCS: https://amitness.com/2019/07/identify-text-language-python/
from collections import Counter
from itertools import chain
from typing import List, Tuple, Union

import numpy as np
from fasttext.FastText import _FastText
from pydash import flatten

from config import config

# Calling _FastText directly, rather than fasttext.load_model() avoids 
# printing warning message
fasttext_model = _FastText(config['fasttext']['model'])


def language_detect_confidence(
    text: Union[str, List[str]],
    threshold=0.5,
    max_chars=10_000,
) -> Tuple[str, float]:
    """
    Results above a threshold of 0.5 are reasonably accurate
    Question: is a zero threshold a better heuristic for segment 
    tokenizers than an empty result?
    DOCS: https://fasttext.cc/docs/en/language-identification.html
    DOCS: https://amitness.com/2019/07/identify-text-language-python/
    :return: ISO 639-1 code
    """
    # text = text.split('\n') if isinstance(text, str) else text
    text = ' '.join(flatten(text)) if not isinstance(text, str) else text
    text = text[:max_chars].replace(r'[^\s]*$', '')  # Performance Optimization for large documents

    # Language identification works better on the whole text rather than
    # on one line at a time. Also, FastText doesn't allow '\n' in input
    text = [ text.replace('\n', ' ') ]

    # Predict the language
    result = fasttext_model.predict(text, k=1, threshold=threshold)  
    try:
        languages    = np.array(list(chain(*result[0]))).flatten()
        confidences  = np.array(list(chain(*result[1]))).flatten()
        # Different lines might have different predictions
        language     = Counter(languages).most_common(1)[0][0]
        # Zero confidence in incorrect predictions
        confidences *= (languages == language)
        confidence   = np.mean(confidences)
    except IndexError:
        language   = ''
        confidence = 0.0
    language = language.replace('__label__', '')
    return language, confidence


def language_detect(text: Union[str, List[str]], threshold=0.5) -> str:
    return language_detect_confidence(text, threshold)[0]

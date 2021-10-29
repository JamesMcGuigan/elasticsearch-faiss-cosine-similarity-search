import glob
from typing import Dict, List, Union

import nltk
from nltk import PunktSentenceTokenizer
from pydash import flatten, flatten_deep

from config import config
from src.utils.fasttest_model import language_detect
from src.utils.files import get_file_rootname, readfile
from src.utils.language_codes import get_language_code


# @clru_cache
def load_punkt_tokenizers() -> Dict[str, PunktSentenceTokenizer]:
    """
    DOCS: https://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.punkt
    """
    punkt_tokenizers = {}
    for filename in glob.glob(config['punkt']['model_glob']):
        lang = get_file_rootname(filename)
        code = get_language_code(lang)
        punkt_tokenizers[code] = nltk.data.load(filename)
    return punkt_tokenizers

punkt_tokenizers = load_punkt_tokenizers()


def get_punkt_tokenizer(lang='', text='') -> PunktSentenceTokenizer:
    def lookups():
        # yield avoids calling expensive functions, if we match on a 
        # simpler case first
        yield lang
        yield get_language_code(lang)
        yield language_detect(text)

    for code in lookups():
        # tokenizer = load_punkt_tokenizer(code)
        # if tokenizer:
        #     return tokenizer
        if code and code in punkt_tokenizers:
            tokenizer = punkt_tokenizers[code]
            break
    else:
        # If we can't figure out the language, then train on the text we 
        # do have, else default to english
        if text:
            tokenizer = PunktSentenceTokenizer()
            try:
                tokenizer.train(text)
            except ValueError:
                tokenizer = punkt_tokenizers['en']
        else:
            tokenizer = punkt_tokenizers['en']
    return tokenizer


def punkt_tokenize_file(filename: str, lang='') -> List[str]:
    return punkt_tokenize_sentences(readfile(filename), lang=lang)

def punkt_tokenize_sentences(text: Union[str, List[str]], lang='') -> List[str]:
    """
    Models: https://www.kaggle.com/nltkdata/punkt
    DOCS:   https://stackoverflow.com/questions/47274540/how-to-improve-sentence-segmentation-of-nltk
    DOCS:   https://www.nltk.org/api/nltk.tokenize.html?highlight=punkt#module-nltk.tokenize.punkt
    Paper:  https://www.aclweb.org/anthology/J06-4003.pdf
    Book:   https://www.nltk.org/book/ch03.html
    """

    # Its quicker to split on '\n' after tokenization than calling Punkt in a loop
    if not isinstance(text, str): "\n".join(flatten(text))
    tokenizer = get_punkt_tokenizer(lang, text)
    sentences = tokenizer.tokenize(text)
    sentences = flatten_deep([ sentence.split('\n') for sentence in sentences ])
    return sentences


def punkt_tokenize_glob(glob_pattern: str) -> Dict[str,List[str]]:
    filenames = glob.glob(glob_pattern)
    return punkt_tokenize_filenames(filenames)

def punkt_tokenize_filenames(filenames: List[str]) -> Dict[str,List[str]]:
    file_sentences = {
        filename: punkt_tokenize_file(filename)
        for filename in filenames
    }
    return file_sentences

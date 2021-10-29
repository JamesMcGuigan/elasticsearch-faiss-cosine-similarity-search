from typing import Any, List


def generate_ngrams( items: List[Any], min_size=1, max_size=4 ) -> List[ List[Any] ]:
    """
    Returns a list of ngram within the min_size, max_size range
    """
    ngrams = [
        items[n:n+size]
        for size in range(min_size, max_size+1)
        for n in range(len(items)-size+1)
    ]
    return ngrams


# def get_ngram_sentences_dict( sentences: List[str], window=4, separator=" " ) -> Dict[Tuple[int], str]:
#     ngrams = {
#         tuple(range(n,n+size)): separator.join(sentences[n:n+size])
#         for size in range(1, window+1)
#         for n in range(len(sentences)-size)
#     }
#     return ngrams

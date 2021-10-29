import hashlib
import os
from typing import Any, Collection, List, Optional

import numpy as np
from humanize import intcomma, naturalsize

from src.utils.batch import batch


class SentenceEncoder:
    """
    Base class providing functionalities for encoding a sentence as a
    fixed-size representation.
    """

    def __init__(self):
        pass

    def __call__(self, sentences: List[str], language: str = None) -> Collection[Any]:
        return self.encode_sentences(sentences, language)

    def encode_sentences(self, sentences: List[str], language: str = None) -> Collection[Any]:
        """ No-op implementation, use str() as the embedding """
        return list(map(str,sentences))

    def encode_sentences_cached(
            self,
            sentences: List[str],
            language: str = None,  # pylint: disable=unused-argument
            batch_size = 10_000,
            verbose=True
    ) -> np.ndarray:
        """ Caches the embedding to the filesystem """
        embedding = self.cache_load_embedding(sentences)
        if embedding is not None:
            return embedding
        else:
            if len(sentences) > batch_size:
                # BUGFIX: CUDA out of memory when trying to encode 2 million sentences
                # Cache large texts in blocks of 10_000, allowing reuse of partial file reads
                embedding = np.concatenate([
                    self.encode_sentences_cached(sentences_batch)
                    for sentences_batch in batch(sentences, batch_size)
                ], axis=0)
                return embedding
            else:
                embedding = np.array(self.encode_sentences(sentences, language))
                self.cache_save_embedding(sentences, embedding, verbose=verbose)
                return embedding

    def cache_hash(self, sentences: List[str]) -> str:
        """ variable length hashing function using shake_256 """
        filename_size = 32  # hash gets truncated to this size
        hash = hashlib.shake_256("\n".join(sentences).encode('utf-8')).hexdigest(filename_size//2)
        return hash

    def cache_filename(self, sentences: List[str]) -> str:
        sha256   = self.cache_hash(sentences)
        filename = f'.cache/{self.__class__.__name__}/{sha256}.npy'
        return filename

    def cache_load_embedding(self, sentences: List[str]) -> Optional[np.ndarray]:
        filename  = self.cache_filename(sentences)
        embedding = None
        if os.path.exists(filename):
            try:
                embedding = np.load(filename)
            except: pass
        return embedding

    def cache_save_embedding(self, sentences: List[str], embedding: np.ndarray, verbose=True) -> None:
        # TODO: alterative method would be to use np.savez() and create a giant { sentence: embedding } dictionary
        filename = self.cache_filename(sentences)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        try:
            np.save(filename, embedding)
        except (KeyboardInterrupt, Exception) as exception:
            os.remove(filename)  # safety catch to prevent incomplete file writes
            raise exception
        if verbose:
            print(f'wrote: {filename} = {naturalsize(os.path.getsize(filename))} = {intcomma(len(embedding))} embeddings')



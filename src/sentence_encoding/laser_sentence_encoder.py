from typing import Any, Callable, Collection, Dict, List

from laserembeddings import Laser

from config import config
from src.sentence_encoding.sentence_encoder import SentenceEncoder
from src.utils.fasttest_model import language_detect


class LaserSentenceEncoder(SentenceEncoder):
    """Class for encoding sentences as sentence embeddings using 
    Facebook's LASER model (https://github.com/facebookresearch/LASER).

    Args:
        bpe_codes (str): Path to LASER's BPE codes 
          (e.g. `93langs.fcodes`).
        bpe_vocab (str): Path to LASER's BPE vocabulary 
          (e.g. `93langs.fvocab`).
        encoder (str): Path to LASER's encoder PyTorch model 
          (e.g. `bilstm.93langs.2018-12-26.pt`)
        tokenizer_options (Dict, optional): Additional arguments to pass 
          to the tokenizer. Defaults to None.
        embedding_options (Dict, optional): Additional arguments to pass
          to the embedding layer. Defaults to None.
        language_detector (Callable, optional): Function that takes a 
          piece of text (string) as the only argument and returns the 
          language in which the text is written. Defaults to None.
    """

    def __init__(
        self,
        bpe_codes: str = config['laser']['bpe_codes'],
        bpe_vocab: str = config['laser']['bpe_vocab'],
        encoder:   str = config['laser']['encoder'],
        tokenizer_options: Dict = None,
        embedding_options: Dict = None,
        language_detector: Callable = language_detect,
    ):
        super().__init__()
        self.laser_model = Laser(
            bpe_codes=bpe_codes,
            bpe_vocab=bpe_vocab, 
            encoder=encoder, 
            tokenizer_options=tokenizer_options, 
            embedding_options=embedding_options
        )
        self.language_detector = language_detector


    def encode_sentences(
        self, 
        sentences: List[str], 
        language: str = None
    ) -> Collection[Any]:
        """Encodes a list of sentences using Facebook's LASER model. 
        Each sentence is transformed into a float vector (embedding).

        Args:
            sentences (List[str]): List of sentences to encode.
            language (str, optional): Language of the sentences. 
              Defaults to None.

        Returns:
            Collection[Any]: 2D NumPy array whose rows represent the 
              encoding for a particular sentence and whose columns 
              represent the different dimensions of the encoding.
        """

        if language is None:
            if self.language_detector is None:
                raise ValueError(
                    '`language_detector` was not specified in the constructor, '
                    'so `language` can\'t be None.'
                )
            # Detect the language in which the sentences are written
            language = self.language_detector(' '.join(sentences))

        return self.laser_model.embed_sentences(sentences, lang=language)

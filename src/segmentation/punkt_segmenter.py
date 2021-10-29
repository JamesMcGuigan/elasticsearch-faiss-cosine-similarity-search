from typing import List

import nltk

from src.segmentation.segmenter import Segmenter
from src.utils.punkt_tokenizer import punkt_tokenize_sentences


class PunktSegmenter(Segmenter):
    """Class for segmenting text documents using NLTK's 
    `PunktSentenceTokenizer`.
    """
    def __init__(self):
        super().__init__()
        self.punkt_sentence_tokenizer = None


    @classmethod
    def from_pretrained_model(cls, path_to_punkt_model: str):
        """Creates a `PunktSegmenter` instance from a pre-trained 
        segmenter.

        Args:
            path_to_punkt_model (str): Path to a pre-trained 
              `nltk.PunktSentenceTokenizer` model.

        Returns:
            PunktSegmenter: A new `PunktSegmenter` instance.
        """

        segmenter = cls()
        segmenter.punkt_sentence_tokenizer = nltk.data.load(
            path_to_punkt_model
        )
        return segmenter


    @classmethod
    def from_text_corpus(cls, text_corpus: str):
        """Creates a `PunktSegmenter` instance by training an 
        `nltk.PunktSentenceTokenizer` model on the given text corpus.

        Args:
            text_corpus (str): Text corpus that will be used to train 
              the segmentation model.

        Returns:
            PunktSegmenter: A new `PunktSegmenter` instance.
        """

        segmenter = cls()
        segmenter.punkt_sentence_tokenizer = nltk.PunktSentenceTokenizer(
            train_text=text_corpus
        )
        return segmenter


    def segment_document(self, document: str) -> List[str]:
        """Segments a document into a list of sentences.

        Args:
            document (str): The text document.

        Returns:
            List[str]: List of sentences found in the document.
        """

        if self.punkt_sentence_tokenizer:
            # BUG: 'PunktSegmenter' object has no attribute 'punkt_sentence_tokenizer' when called via __init__()
            # TODO: auto-detect language
            return self.punkt_sentence_tokenizer.tokenize(document)
        else:
            return punkt_tokenize_sentences(document)

from typing import List

from src.segmentation.segmenter import Segmenter


class NewlineSegmenter(Segmenter):
    """ Segment the document on newlines"""
    def segment_document(self, document: str) -> List[str]:
        return document.split('\n')

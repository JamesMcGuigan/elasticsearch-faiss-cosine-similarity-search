import re
from typing import List

from src.segmentation.segmenter import Segmenter


class PeriodSegmenter(Segmenter):
    """ Segment the document on newlines or periods """
    def segment_document(self, document: str) -> List[str]:
        return re.split(r'[\n.]+', document)

from typing import List

class Segmenter:
    """
    Base class providing functionalities for segmenting a text
    document into individual sentences.
    """

    def segment_document(self, document: str) -> List[str]:
        raise NotImplementedError(
            '`segment_document` has no implementation in class `Segmenter`.'
        )

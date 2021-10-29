from typing import List

from src.segmentation.segmenter import Segmenter

class SeparatorSegmenter(Segmenter):
    """Class for segmenting text documents on specific characters called
    separators (like whitespaces, newlines etc.).

    Args:
        separator (str, optional): The separator used to split text
          documents into individual sentences. Defaults to '\\n'.
    """
    
    def __init__(self, separator='\n'):
        super().__init__()
        self.separator = separator


    def segment_document(self, document: str) -> List[str]:
        """Segments a document into a list of sentences.

        Args:
            document (str): The text document.

        Returns:
            List[str]: List of sentences found in the document.
        """

        return document.split(self.separator)

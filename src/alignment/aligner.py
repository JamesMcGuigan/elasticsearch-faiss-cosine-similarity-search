import sys
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pydash
from pydash import flatten

from src.datastructures.SegmentsAlignment import SegmentsIndexAlignment

# TODO: Mapping = Tuple[int,int] with -1 as None value (for numba static typing)
Mapping = Tuple[
    Optional[int],  # Assuming one-to-one mappings
    Optional[int]
]
Many2ManyMapping = Tuple[
    Union[Optional[int], Sequence[Optional[int]]],
    Union[Optional[int], Sequence[Optional[int]]],
]
Alignment = Sequence[Mapping]
Many2ManyAlignment = Sequence[Many2ManyMapping]

class Aligner:
    """
    Base class providing functionalities for aligning two collections
    of objects. An alignment is a list of mappings between elements in 
    the first collection and elements in the second collection.
    """

    def __init__(self):
        pass

    def align_embeddings(
        self,
        source_objects: List[Any],
        target_objects: List[Any]
    ) -> Alignment:
        raise NotImplementedError(
            '`align_embeddings` has no implementation in class `Aligner`'
        )



    @staticmethod
    def sort_key(mapping: Many2ManyMapping) -> float:
        mapping = flatten(mapping)
        return np.mean([ i for i in mapping if i is not None ]) if len(mapping) else 0.0

    @classmethod
    def cast_one2one_alignment(cls, index_alignment: Union[Alignment, SegmentsIndexAlignment]) -> List[Tuple[int,int]]:
        """
        Expand out a many to many alignment to be a one-to-one alignment
        Also sorts the alignment on the assumption of monotonic alignments
        eg: [ (0,0), (1,[2,3]), ([2,3],3) ] -> [ (0,0),(1,2),(1,3),(2,3),(3,3) ]
        """
        output_alignment = flatten([
            [ ( a, b ) for a in flatten([pair[0]]) for b in flatten([pair[1]]) ]
            for pair in index_alignment
        ])
        # Sort by mean value of tuples, which allows None to be sorted correctly
        # NOTE: this sorting method might break if alignment is not monotonic
        output_alignment = sorted(output_alignment, key=cls.sort_key)
        return (output_alignment)


    @classmethod
    def cast_many2many_alignment(cls, index_alignment: Union[Alignment, SegmentsIndexAlignment]) -> Many2ManyAlignment:
        """
        Expand out a many to many alignment to be a one-to-one alignment
        Also sorts the alignment on the assumption of monotonic alignments
        eg: [ (0,0),(1,2),(1,3),(2,3),(3,3) ] -> [ ([0],[0]), ([1],[2,3]), ([2,3],[3]) ]
        eg: [ (0,0),(1,2),(1,3),(2,None)    ] -> [ ([0],[0]), ([1],[2,3]), ([2],[None]) ]
        """
        index_alignment  = cls.cast_one2one_alignment(index_alignment)  # convert to known integer format
        output_alignment = []
        many2many = (set(),set())
        for alignment in index_alignment:
            # None always get assigned to a new mapping
            if ( alignment[0] in many2many[0] and alignment[0] not in (None, -1)
              or alignment[1] in many2many[1] and alignment[1] not in (None, -1) ):
                many2many[0].add( alignment[0] )
                many2many[1].add( alignment[1] )
            else:
                output_alignment.append( many2many )
                many2many = ( {alignment[0]}, {alignment[1]} )
        output_alignment.append( many2many )

        # cast to Tuple[List[int],List[int]]
        output_alignment = [
            (
                sorted(many2many[0]),
                sorted(many2many[1]),
            )
            for many2many in output_alignment
            if len(many2many[0]) or len(many2many[1])
        ]
        # noinspection PyArgumentList
        return type(index_alignment)(output_alignment)  # return original type


    # TODO: move methods to another file/class if not used by subclasses
    @staticmethod
    def find_all_indexes(text: str, search_string: str, __start=None, __end=None) -> List[int]:
        indexes = []
        try:
            index = 0
            while True:
                index = text.index(search_string, index, __end)  # raises ValueError if not found
                indexes.append(index)
                index += 1  # avoid infinite loop
        except ValueError:
            pass
        return indexes


    # noinspection PyTypeChecker
    @classmethod
    def index_to_char_mapping(cls, segments: List[str], text: str) -> Alignment:
        mapping        = [ [None,None] ] * len(segments)
        end_char_index = 0
        for index, segment in enumerate(segments):
            start_char_index = None
            try:
                start_char_index = text.index(segment, end_char_index)
            except ValueError:
                # Search backwards in the string incase we have accidentally gone too far
                all_indexes = cls.find_all_indexes(text, segment, __end=end_char_index)
                if len(all_indexes):
                    start_char_index = all_indexes[-1]  # last one is closest
                    print(f'index_to_char_mapping(): inconsistent index ' +
                          f'{index} @ {start_char_index} > {end_char_index} for "{segment}"', file=sys.stderr)
                else:
                    print(f'index_to_char_mapping(): unable to find index {index} = "{segment}"', file=sys.stderr)
            if start_char_index is not None:
                end_char_index = start_char_index + len(segment)
                mapping[index] = ( start_char_index, end_char_index )
        return mapping


    # Question: should this function be moved to: src/datastructures/Segments.py ?
    @classmethod
    def convert_index_to_char_alignment(
            cls,
            index_alignment: Union[Alignment,SegmentsIndexAlignment],
            source_segments: List[str],
            target_segments: List[str],
            source_text:     str,
            target_text:     str,
    ) -> Alignment:
        """
        Converts an index-based alignment to a character based alignment
        This method requires access to the original source and target text
        as it cannot be assumed the tokenizer has not stripped whitespace and newlines
        which would mess up character alignment based purely on tokenized segments

        Alignments start at the beginning of the first segment, and at the end of every segment
        TODO: should Alignments be at start and end of every segment, to include whitespace boundaries?
        """

        index_alignment    = cls.cast_many2many_alignment(index_alignment)
        source_index_chars = cls.index_to_char_mapping(source_segments, source_text)
        target_index_chars = cls.index_to_char_mapping(target_segments, target_text)

        char_alignment = []
        for n, index_align in enumerate(index_alignment):
            try:
                index_source_start = index_align[0][0]
                index_target_start = index_align[1][0]
                index_source_end   = index_align[0][-1]
                index_target_end   = index_align[1][-1]

                char_source_start  = source_index_chars[ index_source_start ][0]
                char_target_start  = target_index_chars[ index_target_start ][0]
                char_source_end    = source_index_chars[ index_source_end   ][1]
                char_target_end    = target_index_chars[ index_target_end   ][1]

                align_start = ( char_source_start, char_target_start )
                align_end   = ( char_source_end,   char_target_end   )

                if not None in align_start:
                    char_alignment.append(align_start)
                if not None in align_end:
                    char_alignment.append(align_end)
            except (TypeError,IndexError) as exception:
                pass

        char_alignment = pydash.uniq(char_alignment)
        return char_alignment


    @classmethod
    def convert_index_to_ladder_alignment(
        cls,
        index_alignment: Union[Alignment, SegmentsIndexAlignment]
    ) -> Alignment:
        """
        Converts an index-based alignment to a "ladder" alignment, that
        is an alignment that matches sentence boundaries instead of
        sentence indices.
        """
        ladder_alignment = [(0,0)] if len(index_alignment) else []  # (0, 0) is always present, except for empty input
        prev_index_mapping_type = 'match'
        prev_source_boundary_index = 0
        prev_target_boundary_index = 0
        for source_sentence_index, target_sentence_index in index_alignment:
            if source_sentence_index is None:
                curr_source_boundary_index = prev_source_boundary_index
                curr_target_boundary_index = prev_target_boundary_index + 1
                if prev_index_mapping_type != 'target_unaligned':
                    ladder_alignment.append((
                        curr_source_boundary_index,
                        curr_target_boundary_index
                    ))
                else:
                    ladder_alignment[-1] = (
                        curr_source_boundary_index,
                        curr_target_boundary_index                        
                    )
                prev_index_mapping_type = 'target_unaligned'
            elif target_sentence_index is None:
                curr_source_boundary_index = prev_source_boundary_index + 1
                curr_target_boundary_index = prev_target_boundary_index
                if prev_index_mapping_type != 'source_unaligned':
                    ladder_alignment.append((
                        curr_source_boundary_index,
                        curr_target_boundary_index
                    ))
                else:
                    ladder_alignment[-1] = (
                        curr_source_boundary_index,
                        curr_target_boundary_index                        
                    )
                prev_index_mapping_type = 'source_unaligned'                
            else:
                curr_source_boundary_index = prev_source_boundary_index + 1
                curr_target_boundary_index = prev_target_boundary_index + 1
                ladder_alignment.append((
                    curr_source_boundary_index,
                    curr_target_boundary_index
                ))
                prev_index_mapping_type = 'match'

            prev_source_boundary_index = curr_source_boundary_index
            prev_target_boundary_index = curr_target_boundary_index
        
        return ladder_alignment

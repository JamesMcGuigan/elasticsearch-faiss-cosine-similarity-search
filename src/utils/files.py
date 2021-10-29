#!/usr/bin/env python3
"""
Workflow is to the take the Universal Declaration of Human Rights (UDHR) 
corpus in 372 languages
- Use and train PunktSentenceTokenizer() for sentence segmentation
- Compare the number of sentences generated for each translation
- Compute LASER and LaBSE embeddings for each sentence
- Attempt to figure out a matching algorithm based on cosine similarity
- Output to XLIFF format
"""
import glob
import os
import sys
from itertools import combinations
from typing import List, Tuple, Union

import ftfy
import pydash
from pydash import flatten


def file_readlines(filename: str, count=0) -> List[str]:
    if count <= 0: count = sys.maxsize
    try:
        with open(filename, 'r') as file:
            lines = []
            while line := file.readline():
                lines.append(line.strip('\n').strip(' '))
                if len(lines) >= count: break
    except UnicodeDecodeError:
        # BUGFIX: 'datasets/europarl/europarl/txt/pl/ep-09-10-22-009.txt' throws UnicodeDecodeError
        text  = readfile(filename)
        lines = text.split('\n')[:count]
    return lines


def read_lines(file_path: str, max_lines=sys.maxsize):
    with open(file_path, mode='r', encoding='utf-8') as file_pointer:
        lines = file_pointer.read().split('\n')[:max_lines]
        return lines


def readfile(filename: str) -> str:
    try:
        with open(filename, 'r') as file:
            text = file.read()
    except UnicodeDecodeError:
        # BUGFIX: 'datasets/europarl/europarl/txt/pl/ep-09-10-22-009.txt' throws UnicodeDecodeError
        with open(filename, 'rb') as file:
            bytes = file.read()
            text  = ftfy.guess_bytes(bytes)[0]
            text  = ftfy.fix_text(text)
    return text


def savefile(text: Union[str,List[str]], filename: str) -> None:
    if os.path.dirname(filename):  # BUGFIX: './filename.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        file.write(text)


def get_file_rootname(filename: str) -> str:
    return os.path.basename(filename).split('.')[0]

def get_file_extension(filename: str) -> str:
    return os.path.basename(filename).split('.')[-1]


def get_file_pairs(glob_path: str) -> List[Tuple[str,str]]:
    """
    Return all unique combinations of matching file pairs within a dataset
    Europarl files return [ ('lang1-lang2/lang1.txt', 'lang1-lang2/lang2.txt'), ... ]
    TMX      files return [ ('dir/filename', ''), ... ]
    UDHR and Europarl_Samples may return very long lists of all language pair permutations
    """
    size = 1 if '.tmx' in glob_path else 2
    filenames = glob.glob(glob_path)
    groupings = pydash.group_by(filenames, os.path.dirname)
    for key, files in groupings.items():
        if size == 1:
            groupings[key] = [ (file,'') for file in groupings[key] ]
        else:
            groupings[key] = list(combinations(files, size))
    file_pairs = flatten(list(groupings.values()))
    return file_pairs

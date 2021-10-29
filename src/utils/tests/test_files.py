import os

import pytest

from config import config
from src.utils.files import get_file_pairs, readfile


@pytest.mark.parametrize('filename', [
    'datasets/udhr/txt/eng.txt',                                # Happy Path
    'datasets/europarl_raw/txt/pl/ep-09-10-22-009.txt',         # UnicodeDecodeError
    # *glob.glob(config['datasets']['udhr']['input_glob']),     # Works but very slow
    # *glob.glob(config['datasets']['europarl']['input_glob'])  # Works but very slow
])
def test_readfile(filename):
    if os.path.exists(filename):
        text = readfile(filename)
        # text = open(filename).read()  # confirm bug
        assert type(text) == str
        assert len(text)


@pytest.mark.parametrize('glob_path', [
    config['datasets']['udhr']['input_glob'],
    config['datasets']['europarl']['input_glob'],
    config['datasets']['europarl_samples']['input_glob'],
    config['datasets']['tmx_alignment_tool']['input_glob'],
])
def test_get_file_pairs(glob_path):
    file_pairs = get_file_pairs(glob_path)
    assert len(file_pairs)
    assert len(file_pairs) == len(set(map(frozenset,file_pairs)))  # list should contain no pairwise duplicates
    assert isinstance(file_pairs,list)
    for file_pair in file_pairs:
        assert len(file_pair) == 2
        assert isinstance(file_pair[0], str)
        assert isinstance(file_pair[1], str)
        assert file_pair[0] != ''
        if '.tmx' not in glob_path: assert file_pair[1] != ''
        else:                       assert file_pair[1] == ''

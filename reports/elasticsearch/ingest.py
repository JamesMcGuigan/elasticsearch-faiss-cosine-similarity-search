#!/usr/bin/env python3
# Idea:
# - create a CLI or possibly web interface
# - Allow user to type in a sentence, or sentence extracted from document
# - Create a laser embedding for it
# - Query ElasticSearch and ask it to return the closest cosine similarity sentences in the desired language
# - Could possibly also try searching for the sentence in the source language, using ElasticSearch TF-IDF scoring, and using the precached embedding for those sentences as lookup
# - Database could possibly be extended with comma segmented sentences or phrase ngrams (Punkt sent_end_chars = ('.', '?', '!'))

import glob
import hashlib
import os
import sys
from typing import Any, Iterable, Set

import pydash
from elasticsearch.helpers import bulk
from humanize import intcomma
from pydash import py_, uniq

from config import config
from reports.elasticsearch.client import es
from src.utils.batch import batch
from src.utils.fasttest_model import language_detect
from src.utils.files import readfile
from src.utils.language_codes import get_language_name
from src.utils.laser_encode import laser_encode
from src.utils.punkt_tokenizer import punkt_tokenize_sentences
from src.utils.tuplize import tuplize


def get_missing_ids(ids: Iterable[Any]) -> Set[Any]:
    ids = list(ids)
    query = {
        "_source": [ "_id" ],
        "size":    len(ids),     # return all, defaults to 10
        "query":   { "ids": {"values": list(ids) } }
    }
    response = es.search(query, os.getenv('INDEX'))
    existing = py_(response).get('hits.hits').pluck('_id').value()
    missing  = set(ids) - set(existing)
    return missing


# TODO: make async or use pathos.multiprocessing
def ingest_file(file, dataset, min_length=5, batch_size=10000, min_lines=0, max_lines=0, verbose=True):
    stats = { "success": 0, "errors": 0, "cached": 0 }
    text  = readfile(file)
    if min_lines or max_lines:
        text = "\n".join( text.split('\n')[ min_lines or 0 : max_lines or sys.maxsize ] )

    lang_code  = language_detect(text)
    language   = get_language_name(lang_code)
    # sentences  = text.split('\n')
    sentences  = punkt_tokenize_sentences(text, lang=lang_code)
    sentences  = uniq(filter(lambda s: len(s) > min_length, map(lambda s: s.strip(), sentences)))

    if verbose: print(f'{file} ({language}) = {intcomma(len(sentences))} sentences')
    for sentences_batch in batch(sentences, batch_size):
        md5s    = [ hashlib.md5(sentence.encode('utf-8')).hexdigest() for sentence in sentences_batch ]
        missing = get_missing_ids(md5s)
        stats['cached'] += len(set(md5s)) - len(missing)
        if len(missing) == 0: continue
        md5s, sentences_batch = zip(*[
            (md5, sentence)
            for (md5, sentence) in zip(md5s, sentences_batch)
            if md5 in missing
        ])
        embeddings = laser_encode(sentences_batch, lang_code)

        # DOCS: https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html
        bulk_actions = [
            {
                '_op_type': 'index',
                '_index': os.getenv('INDEX'),
                '_id':    md5,  # unique primary key
                '_source': {
                    "text":            sentence,
                    "length":          len(sentence),
                    "lang_code":       lang_code,
                    "language":        language,
                    "dataset":         dataset,
                    "file":            file,
                    "embedding_laser": embedding,
                }
            }
            for sentence, md5, embedding in zip(sentences_batch, md5s, embeddings)
        ]
        response = list(bulk(
            es, bulk_actions,
            max_retries=5,
            initial_backoff=2,
            raise_on_error=True,
            stats_only=False
        ))
        stats['success'] += response[0]
        stats['errors']  += len(response[1])
        if len(response[1]): print('ES Errors: ', set(tuplize(pydash.pluck(response[1], 'index.error'))))
        if verbose: print(stats)


if __name__ == '__main__':
    dataset = "europarl"
    files   = glob.glob(config['datasets']['europarl']['input_glob'])
    # files   = list(glob.glob("datasets/europarl/it-en/*"))
    print(files)
    for min_lines, max_lines in [ (0,100_000), (100_000,500_000), (500_000,1_000_000), (1_000_000,0) ]:
        for file in files:
            ingest_file(file, dataset, min_lines=min_lines, max_lines=max_lines)

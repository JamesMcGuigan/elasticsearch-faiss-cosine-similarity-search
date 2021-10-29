#!/usr/bin/env python3
# TODO: implement fast cosine similarity: https://github.com/StaySense/fast-cosine-similarity
# TODO: investigate KNN search: https://docs.aws.amazon.com/elasticsearch-service/latest/developerguide/knn.html
#       KNN requires AWS or ES opendistro - https://opendistro.github.io/for-elasticsearch-docs/docs/knn/
# Examples:
# reports/elasticsearch/query.py -v -t "Madam President, on a point of order." -c 10
# reports/elasticsearch/query.py -v -t "Four Score and Seven Years Ago"
# reports/elasticsearch/query.py -v -t "I should like to observe a minute's silence" --lang italian

import argparse
import os
import time

from pydash import get

from reports.elasticsearch.client import es
from src.utils.language_codes import get_language_name
from src.utils.laser_encode import laser_encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text',      help='text to query',  type=str, required=True)
    # parser.add_argument('-e', '--embedding', help='embedding type', type=str, default='laser')
    parser.add_argument('-l', '--lang',      help='desired language', type=str)
    parser.add_argument('-c', '--count',     help='number of results', type=int, default=10)
    parser.add_argument('-v', '--verbose',   help='show timings', action='store_true')

    args = parser.parse_args()
    results = query_embedding(text=args.text, lang=args.lang, count=args.count, verbose=args.verbose)
    for result in results:
        print(f"{result['score']:.6f}\t{result['language']:10s}\t{result['text']}")


# DOCS: https://www.elastic.co/blog/text-similarity-search-with-vectors-in-elasticsearch
# DOCS: https://github.com/jtibshirani/text-embeddings/blob/master/src/main.py
def query_embedding(text, lang=None, count=10, verbose=True):
    time_start = {}
    time_end   = {}
    time_start["embedding"] = time.perf_counter()
    embedding_key = f'embedding_laser'
    embedding     = laser_encode([ text ])  # pass in list to prevent punkt tokenization
    time_end["embedding"] = time.perf_counter()

    query = {
        # "explain": True,
        # "profile": True,
        "size": count,
        "_source": [ "_score", "text", "length", "lang_code", "language", "dataset", "file" ],
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                # We make sure to pass the query vector as a script parameter to avoid recompiling the script() on every new query.
                "script": {
                    # Since Elasticsearch does not allow negative scores, it's necessary to add one to the cosine similarity.
                    "source": f"cosineSimilarity(params.query_vector, '{embedding_key}') + 1.0",
                    "params": { "query_vector": embedding.tolist()[0] }
                }
            }
        }
    }
    if lang:
        query['query']['script_score']['query'] = {
            "bool": {
                "should": [
                    { "match": { "lang_code": lang.lower() } },
                    { "match": { "language":  lang.lower().capitalize() } },  # keyword field is case insensitive
                    { "match": { "language":  get_language_name(lang)   } },  # function used by ingest
                ]
            }
        }

    time_start["query"] = time.perf_counter()
    response = es.search(query, os.getenv('INDEX'))
    time_end["query"] = time.perf_counter()

    results  = []
    for hit in get(response, 'hits.hits', []):
        results += [{
            "score":    get(hit, '_score', 0.0) - 1.0,  # Remove +1.0 that was added to the query score
            "language": get(hit, '_source.language', ''),
            "text":     get(hit, '_source.text', ''),
            # "file":     get(hit, '_source.file', ''),
        }]

    time_taken = { key: round(time_end[key] - time_start[key],3) for key in time_end.keys() }
    if verbose: print("query_embedding() timings:", time_taken)

    return results


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Tensorflow INFO/WARNING messages are not printed
    os.environ['CUDA_VISIBLE_DEVICES'] = ''   # disable GPU - faster load time for small documents
    main()

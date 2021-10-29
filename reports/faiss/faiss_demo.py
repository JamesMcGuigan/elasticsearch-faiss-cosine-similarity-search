#!/usr/bin/env python3
# DOCS: https://github.com/facebookresearch/faiss/wiki/Getting-started
# Paper: https://arxiv.org/pdf/1702.08734.pdf - Billion-scale similarity search with GPUs


### CPU vs GPU
### 500,000 lines is the largest that can fit in GPU memory without PCA
### GTX-1600 GPU is about 35x faster than i7-7700HQ CPU @ 2.80GHz
### CPU + GPU have the same accuracy
### Indexing time scales O(N) | Search time for N indexes scales O(N^2), so single search is linear with dataset size
### TODO: embeddings should be normalized before caching and not counted as part of index time

# cpu | accuracy: 0.96 | clusters: 1/1 | pca: 0 | lines: 100,000 | time: {'io':  9.589, 'faiss_index': 0.316, 'faiss_search':  238.871, 'stats': 0.354}
# cpu | accuracy: 0.95 | clusters: 1/1 | pca: 0 | lines: 250,000 | time: {'io': 16.053, 'faiss_index': 0.692, 'faiss_search': 1491.449, 'stats': 0.737}

# gpu | accuracy: 0.96 | clusters: 1/1 | pca: 0 | lines: 100,000 | time: {'io':  9.749, 'faiss_index': 0.272, 'faiss_search':    6.982, 'stats': 0.312}
# gpu | accuracy: 0.95 | clusters: 1/1 | pca: 0 | lines: 250,000 | time: {'io': 15.944, 'faiss_index': 0.43,  'faiss_search':   42.272, 'stats': 0.678}
# gpu | accuracy: 0.95 | clusters: 1/1 | pca: 0 | lines: 500,000 | time: {'io': 37.355, 'faiss_index': 0.743, 'faiss_search':  187.633, 'stats': 1.538}


### PCA
### This allows us to reduce memory usage to fit larger datasets into GPU memory
### PCA reduces accuracy, but 128-256 dimentions still gives reasonable results
### PCA with 1024 dimensions scores slightly worse than original 1024 dimensions without PCA
### --cutoff unaligned_score has different statistics with PCA | accuracy doesn't implement cutoff
###          in theory cosine_similarity of pre-PCA embeddings could be double checked after search
### faiss_index:  scales O(log(N)) + 3.5s added to indexing time relative to PCA dimention size
### faiss_search: scales O(N^1.5) relative to PCA dimension size
# gpu | accuracy: 0.40 | cutoff: 0.25 | precision: 0.09 | recall: 0.01 | f1: 0.02 | clusters: 1/1 | pca:   32 | lines:   100,000 | time: {'io':   0.980, 'faiss_index':  3.545, 'faiss_search':   0.840, 'stats': 0.882 }
# gpu | accuracy: 0.79 | cutoff: 0.25 | precision: 0.45 | recall: 0.09 | f1: 0.16 | clusters: 1/1 | pca:   64 | lines:   100,000 | time: {'io':   0.987, 'faiss_index':  3.618, 'faiss_search':   0.953, 'stats': 0.810 }
# gpu | accuracy: 0.90 | cutoff: 0.25 | precision: 0.78 | recall: 0.30 | f1: 0.44 | clusters: 1/1 | pca:  128 | lines:   100,000 | time: {'io':   0.992, 'faiss_index':  3.713, 'faiss_search':   1.600, 'stats': 0.837 }
# gpu | accuracy: 0.93 | cutoff: 0.25 | precision: 0.91 | recall: 0.68 | f1: 0.78 | clusters: 1/1 | pca:  256 | lines:   100,000 | time: {'io':   1.008, 'faiss_index':  4.216, 'faiss_search':   2.529, 'stats': 0.833 }
# gpu | accuracy: 0.94 | cutoff: 0.25 | precision: 0.93 | recall: 0.87 | f1: 0.90 | clusters: 1/1 | pca:  512 | lines:   100,000 | time: {'io':   1.053, 'faiss_index':  4.824, 'faiss_search':   4.609, 'stats': 0.864 }
# gpu | accuracy: 0.94 | cutoff: 0.25 | precision: 0.94 | recall: 0.90 | f1: 0.92 | clusters: 1/1 | pca: 1024 | lines:   100,000 | time: {'io':   0.999, 'faiss_index':  6.209, 'faiss_search':   9.018, 'stats': 0.889 }
# gpu | accuracy: 0.96 | cutoff: 0.75 | precision: 0.96 | recall: 0.93 | f1: 0.94 | clusters: 1/1 | pca:    0 | lines:   100,000 | time: {'io':   0.958, 'faiss_index':  0.263, 'faiss_search':   6.662, 'stats': 0.903 }
                                                                                                                         
### PCA with 256 dimensions is 2.3x quicker                                                                              
# gpu | accuracy: 0.93 | cutoff: 0.20 | precision: 0.92 | recall: 0.89 | f1: 0.91 | clusters: 1/1 | pca:  256 | lines:   100,000 | time: {'io':   5.227, 'faiss_index':  7.294, 'faiss_search':   2.995, 'stats': 0.963 }
# gpu | accuracy: 0.92 | cutoff: 0.20 | precision: 0.91 | recall: 0.88 | f1: 0.90 | clusters: 1/1 | pca:  256 | lines:   250,000 | time: {'io':  15.835, 'faiss_index':  9.111, 'faiss_search':  17.942, 'stats': 2.422 }
# gpu | accuracy: 0.91 | cutoff: 0.20 | precision: 0.91 | recall: 0.88 | f1: 0.89 | clusters: 1/1 | pca:  256 | lines:   500,000 | time: {'io':  51.096, 'faiss_index': 12.276, 'faiss_search':  68.579, 'stats': 4.877 }
                                                                                                                                                                                                                        
### PCA with 128 dimensions is 4x quicker                                                                                
### Makes it possible to do BiText retrevial on a 2 million line document in 4 + 12 = 16 minutes
### Search times comparable IO times to load cached embeddngs from a cold HDD (without memory caching)
# gpu | accuracy: 0.90 | cutoff: 0.20 | precision: 0.87 | recall: 0.65 | f1: 0.75 | clusters: 1/1 | pca:  128 | lines:   100,000 | time: {'io':   9.195, 'faiss_index':   4.152, 'faiss_search':   1.846, 'stats': 0.947 }
# gpu | accuracy: 0.88 | cutoff: 0.20 | precision: 0.86 | recall: 0.65 | f1: 0.74 | clusters: 1/1 | pca:  128 | lines:   250,000 | time: {'io':  11.245, 'faiss_index':   7.096, 'faiss_search':  10.437, 'stats': 2.444 }
# gpu | accuracy: 0.87 | cutoff: 0.20 | precision: 0.85 | recall: 0.65 | f1: 0.74 | clusters: 1/1 | pca:  128 | lines:   500,000 | time: {'io':  41.796, 'faiss_index':  15.442, 'faiss_search':  41.970, 'stats': 4.394 }
# gpu | accuracy: 0.87 | cutoff: 0.20 | precision: 0.85 | recall: 0.68 | f1: 0.76 | clusters: 1/1 | pca:  128 | lines: 1,000,000 | time: {'io': 116.957, 'faiss_index':  23.362, 'faiss_search': 163.236, 'stats': 9.674 }
# gpu | accuracy: 0.86 | cutoff: 0.20 | precision: 0.84 | recall: 0.68 | f1: 0.76 | clusters: 1/1 | pca:  128 | lines: 1,920,209 | time: {'io': 298.077, 'faiss_index': 239.149, 'faiss_search': 720.527, 'stats': 39.577}


### Clusers
### index time scales O(N^2) with number of embeddings and O(N^1.35) with number of clusters
### accuracy reduces --nlist, but increases with --nprobe
### clustering is inefficient for single search bitext retrieval
### search time can be faster for small --nprobe, but slower for large --nprobe
### search time scales ~sqrt(N) with --nprobe
### search time scales O(1/N^0.7) with number of clusters
### large unexplained mismatch between accuracy and precision - disable --cutoff

# gpu | accuracy: 0.64 | cutoff: 0.75 | precision: 0.01 | recall: 0.00 | f1: 0.00 | clusters:  1/1562 | pca: 0 | lines: 100,000 | time: {'io':  1.061, 'faiss_index':  51.064, 'faiss_search':  3.384, 'stats': 0.912}
# gpu | accuracy: 0.77 | cutoff: 0.75 | precision: 0.02 | recall: 0.00 | f1: 0.00 | clusters:  2/1562 | pca: 0 | lines: 100,000 | time: {'io':  0.720, 'faiss_index':  37.637, 'faiss_search':  3.682, 'stats': 0.795}
# gpu | accuracy: 0.89 | cutoff: 0.75 | precision: 0.11 | recall: 0.00 | f1: 0.00 | clusters:  8/1562 | pca: 0 | lines: 100,000 | time: {'io':  9.944, 'faiss_index':  43.036, 'faiss_search':  8.660, 'stats': 1.03}
# gpu | accuracy: 0.94 | cutoff: 0.75 | precision: 0.23 | recall: 0.00 | f1: 0.00 | clusters: 40/1562 | pca: 0 | lines: 100,000 | time: {'io':  0.828, 'faiss_index':  40.073, 'faiss_search': 30.087, 'stats': 0.791}
# gpu | accuracy: 0.88 | cutoff: 0.75 | precision: 0.06 | recall: 0.00 | f1: 0.00 | clusters:  9/3906 | pca: 0 | lines: 250,000 | time: {'io': 24.179, 'faiss_index': 246.755, 'faiss_search': 25.468, 'stats': 2.751}

# gpu | accuracy: 0.78 | cutoff: 0.75 | precision: 0.00 | recall: 0.00 | f1: 0.00 | clusters: 1/32   | pca: 0 | lines: 100,000 | time: {'io': 1.052, 'faiss_index':  0.575, 'faiss_search':  64.224, 'stats': 0.835}
# gpu | accuracy: 0.72 | cutoff: 0.75 | precision: 0.01 | recall: 0.00 | f1: 0.00 | clusters: 1/64   | pca: 0 | lines: 100,000 | time: {'io': 1.247, 'faiss_index':  1.467, 'faiss_search':  39.240, 'stats': 0.83}
# gpu | accuracy: 0.68 | cutoff: 0.75 | precision: 0.01 | recall: 0.00 | f1: 0.00 | clusters: 1/128  | pca: 0 | lines: 100,000 | time: {'io': 1.114, 'faiss_index':  1.767, 'faiss_search':  21.495, 'stats': 0.84}
# gpu | accuracy: 0.65 | cutoff: 0.75 | precision: 0.01 | recall: 0.00 | f1: 0.00 | clusters: 1/256  | pca: 0 | lines: 100,000 | time: {'io': 1.106, 'faiss_index':  5.451, 'faiss_search':  14.339, 'stats': 0.894}
# gpu | accuracy: 0.64 | cutoff: 0.75 | precision: 0.01 | recall: 0.00 | f1: 0.00 | clusters: 1/512  | pca: 0 | lines: 100,000 | time: {'io': 1.135, 'faiss_index': 14.096, 'faiss_search':   8.694, 'stats': 0.858}
# gpu | accuracy: 0.63 | cutoff: 0.75 | precision: 0.01 | recall: 0.00 | f1: 0.00 | clusters: 1/1024 | pca: 0 | lines: 100,000 | time: {'io': 1.053, 'faiss_index': 27.555, 'faiss_search':   5.164, 'stats': 0.996}

# gpu | accuracy: 0.89 | cutoff: 0.75 | precision: 0.05 | recall: 0.00 | f1: 0.00 | clusters: 2/32   | pca: 0 | lines: 100,000 | time: {'io': 1.086, 'faiss_index':  0.650, 'faiss_search': 104.516, 'stats': 0.840}
# gpu | accuracy: 0.85 | cutoff: 0.75 | precision: 0.03 | recall: 0.00 | f1: 0.00 | clusters: 2/64   | pca: 0 | lines: 100,000 | time: {'io': 1.041, 'faiss_index':  1.060, 'faiss_search':  62.361, 'stats': 0.973}
# gpu | accuracy: 0.82 | cutoff: 0.75 | precision: 0.03 | recall: 0.00 | f1: 0.00 | clusters: 2/128  | pca: 0 | lines: 100,000 | time: {'io': 1.102, 'faiss_index':  1.841, 'faiss_search':  36.658, 'stats': 0.894}
# gpu | accuracy: 0.79 | cutoff: 0.75 | precision: 0.01 | recall: 0.00 | f1: 0.00 | clusters: 2/256  | pca: 0 | lines: 100,000 | time: {'io': 1.091, 'faiss_index':  5.190, 'faiss_search':  21.002, 'stats': 0.885}
# gpu | accuracy: 0.78 | cutoff: 0.75 | precision: 0.03 | recall: 0.00 | f1: 0.00 | clusters: 2/512  | pca: 0 | lines: 100,000 | time: {'io': 1.093, 'faiss_index': 14.283, 'faiss_search':  11.427, 'stats': 0.861}
# gpu | accuracy: 0.77 | cutoff: 0.75 | precision: 0.03 | recall: 0.00 | f1: 0.00 | clusters: 2/1024 | pca: 0 | lines: 100,000 | time: {'io': 1.323, 'faiss_index': 26.462, 'faiss_search':   6.157, 'stats': 0.914}

# gpu | accuracy: 0.94 | cutoff: 0.75 | precision: 0.09 | recall: 0.00 | f1: 0.00 | clusters: 4/32   | pca: 0 | lines: 100,000 | time: {'io': 1.117, 'faiss_index':  0.626, 'faiss_search': 147.598, 'stats': 0.81}
# gpu | accuracy: 0.92 | cutoff: 0.75 | precision: 0.08 | recall: 0.00 | f1: 0.00 | clusters: 4/64   | pca: 0 | lines: 100,000 | time: {'io': 1.119, 'faiss_index':  0.870, 'faiss_search':  90.712, 'stats': 0.798}
# gpu | accuracy: 0.90 | cutoff: 0.75 | precision: 0.07 | recall: 0.00 | f1: 0.00 | clusters: 4/128  | pca: 0 | lines: 100,000 | time: {'io': 1.140, 'faiss_index':  1.955, 'faiss_search':  53.235, 'stats': 1.129}
# gpu | accuracy: 0.87 | cutoff: 0.75 | precision: 0.03 | recall: 0.00 | f1: 0.00 | clusters: 4/256  | pca: 0 | lines: 100,000 | time: {'io': 1.557, 'faiss_index':  5.794, 'faiss_search':  30.869, 'stats': 0.758}
# gpu | accuracy: 0.86 | cutoff: 0.75 | precision: 0.04 | recall: 0.00 | f1: 0.00 | clusters: 4/512  | pca: 0 | lines: 100,000 | time: {'io': 0.990, 'faiss_index': 12.613, 'faiss_search':  16.996, 'stats': 0.759}
# gpu | accuracy: 0.85 | cutoff: 0.75 | precision: 0.05 | recall: 0.00 | f1: 0.00 | clusters: 4/1024 | pca: 0 | lines: 100,000 | time: {'io': 1.079, 'faiss_index': 26.485, 'faiss_search':   8.563, 'stats': 0.744}


import argparse
import gc
import glob
import math
import os
import time
from itertools import chain
from typing import List, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Tensorflow INFO/WARNING messages are not printed
# os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())

import faiss
import numpy as np
from humanize import intcomma

from src.alignment_scoring.precision_recall_f1_scorer import precision_recall_f1_score
from src.sentence_encoding.labse_sentence_encoder import LabseSentenceEncoder
from src.sentence_encoding.laser_sentence_encoder import LaserSentenceEncoder
from src.utils.files import file_readlines


def accuracy_score(distance_indexes, target_sentences, depth=1) -> float:
    correct = 0
    for index_source in range(len(distance_indexes)):
        target_indexes = distance_indexes[index_source][:depth]
        if index_source in target_indexes:
            correct += 1

        elif target_sentences[index_source].strip() in [
            target_sentences[target_index].strip()
            for target_index in target_indexes
        ]:
            correct += 1

    accuracy = correct / len(distance_indexes)
    return accuracy


def faiss_index_alignment(distance_indexes, distance_scores, unaligned_score=0.5) -> List[Tuple[int, int]]:
    index_alignment = []
    for source_index in range(len(distance_indexes)):
        target_index = None
        if distance_scores[source_index][0] >= unaligned_score:
            target_index = distance_indexes[source_index][0]
        index_alignment.append( (source_index, target_index) )
    return index_alignment


encoders = {
    "laser": LaserSentenceEncoder,
    "labse": LabseSentenceEncoder,
}

if __name__ == '__main__':
    time_start = {}
    time_end   = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  type=str, default='datasets/europarl/de-en/europarl-v7.de-en.en')
    parser.add_argument("--target",  type=str, default='datasets/europarl/de-en/europarl-v7.de-en.de')
    parser.add_argument("--encoder", type=str, default='laser', help='laser | labse')

    parser.add_argument("--gpu",     action='store_true')
    parser.add_argument("--lines",   type=int, default=0)
    parser.add_argument("--batch",   type=int, default=10_000, help='batch size for faiss search | 10,000 is optimal')
    parser.add_argument("--depth",   type=int, default=1,      help='how many candidates per sentence')
    parser.add_argument("--pca",     type=int, default=0,      help='size of PCA embedding reduction')
    parser.add_argument("--cluster", action='store_true',      help='enable fast clustering')
    parser.add_argument("--nlist",   type=int, default=0,      help='# of clusters to create | default = 64 nodes per cluster')
    parser.add_argument("--nprobe",  type=int, default=0,      help='# of clusters to search when clustering | default = log(nlist)')
    parser.add_argument("--cutoff",  type=float, default=0.75, help='unaligned_score | disable for PCA')

    args = parser.parse_args()
    if not args.gpu: os.environ['CUDA_VISIBLE_DEVICES'] = ''   # disable GPU - faster load time for small documents

    encoder = encoders[args.encoder]()

    time_start['io']  = time.perf_counter()

    source_sentences  = list(chain(*[ file_readlines(filename, args.lines) for filename in glob.glob(args.source) ]))
    target_sentences  = list(chain(*[ file_readlines(filename, args.lines) for filename in glob.glob(args.target) ]))
    source_embeddings = encoder.encode_sentences_cached(source_sentences)  # casting to float16 breaks FAISS
    target_embeddings = encoder.encode_sentences_cached(target_sentences)

    time_end['io']    = time.perf_counter()
    gc.collect()

    # # Cosine Similarity scales O(N^2) for both CPU and memory | 50,000 lines = 10Gb RAM
    # time_start['cosine_similarity'] = time.perf_counter()
    # cosine_similarity(source_embeddings, target_embeddings)
    # time_end['cosine_similarity'] = time.perf_counter()

    time_start['faiss_index'] = time.perf_counter()

    # Cosine Similarity Distance using matmul requires normalizing embeddings first
    # TODO: embeddings should be normalized before caching and not counted as part of index time
    faiss.normalize_L2(source_embeddings)
    faiss.normalize_L2(target_embeddings)

    target_index = faiss.IndexFlat(args.pca or target_embeddings.shape[1], faiss.METRIC_INNER_PRODUCT)

    # Reduce dimensionality to load larger datasets into GPU memory
    if args.pca:
        # DOCS: https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint
        # ERROR: 'nbits_per_idx <= 8' failed | can't get this method to work
        # assert target_embeddings.shape[1] % args.pca == 0  # embedding size must be a multiple of args.pca
        # target_index = faiss.IndexIVFPQ(target_index, target_embeddings.shape[1], nlist, args.subquantizers, args.pca)
        # target_index.nprobe = args.nprobe

        # DOCS: https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing
        pca_matrix   = faiss.PCAMatrix(target_embeddings.shape[1], args.pca, eigen_power=0, random_rotation=False)
        target_index = faiss.IndexPreTransform(pca_matrix, target_index)
        target_index.train(target_embeddings)


    nlist  = 1
    nprobe = 1
    if args.cluster:
        # DOCS: https://github.com/facebookresearch/faiss/wiki/Faster-search
        # nlist affects clustering, minimum is 39x samples per cluster - requires further investigation
        # nprobe affects
        nlist  = args.nlist  or len(target_embeddings) // 64       # TODO: optimize guess
        nprobe = args.nprobe or math.ceil( math.log( nlist ) )     # TODO: optimize guess
        # nprobe = args.nprobe or math.ceil( math.sqrt( nlist ) )  # TODO: optimize guess
        target_index = faiss.IndexIVFFlat(target_index, target_embeddings.shape[1], nlist)
        target_index.train(target_embeddings)
        target_index.nprobe = nprobe  # how many clusters to search

    if args.gpu:
        # DOCS: https://github.com/facebookresearch/faiss/wiki/Running-on-GPUs
        # GPU can handle a search index of max size <250,000
        res = faiss.StandardGpuResources()  # use a single GPU
        target_index = faiss.index_cpu_to_gpu(res, 0, target_index)


    target_index.add(target_embeddings)

    time_end['faiss_index'] = time.perf_counter()
    # Batch size of 10,000 is optimal
    # 50,000 lines:  {'io': 136.674, 'faiss_index': 0.09, 'faiss_search_all': 66.999, 'faiss_search_batch_1000': 67.683, 'faiss_search_batch_10000': 51.484}
    time_start['faiss_search'] = time.perf_counter()

    distance_scores  = []
    distance_indexes = []
    for n in range(0, len(source_embeddings), args.batch):
        distance_score, distance_index = target_index.search(source_embeddings[n:n + args.batch], args.depth)
        distance_scores.append(distance_score)
        distance_indexes.append(distance_index)
    distance_scores  = np.concatenate(distance_scores,  axis=0)
    distance_indexes = np.concatenate(distance_indexes, axis=0)

    time_end['faiss_search'] = time.perf_counter()
    time_start['stats']      = time.perf_counter()


    ground_truth_alignment      = [ (index, index) if sentence else (index, None)
                                    for index, sentence in enumerate(source_sentences) ]
    index_alignment             = faiss_index_alignment(distance_indexes, distance_scores, args.cutoff)
    precision, recall, f1_score = precision_recall_f1_score(index_alignment, ground_truth_alignment)
    for depth in range(1, args.depth+1):
        accuracy = accuracy_score(distance_indexes, target_sentences)
        # print(f"depth: {depth:2d} = {accuracy:.2f} accuracy")

    time_end[f'stats'] = time.perf_counter()
    print(('gpu' if args.gpu else 'cpu'),
          f'| accuracy: {accuracy:.2f}',
          f'| cutoff: {args.cutoff:.2f}',
          f'| precision: {precision:.2f}',
          f'| recall: {recall:.2f}',
          f'| f1: {f1_score:.2f}',
          f'| clusters: {nprobe}/{nlist}',
          f'| pca: {args.pca}',
          f'| lines:', intcomma(len(target_sentences)),
          f'| time:', { key: round(time_end[key] - time_start[key],3) for key in time_end.keys() }
    )

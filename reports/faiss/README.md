# FAISS

## Findings
[./faiss_demo.py](faiss_demo.py)

- FAISS runs 35x faster on GPU than CPU
- FAISS scales linearly with search space and number of queries | O(N^2) for bitext retrieval
- GTX 1600 (6GB) can load 500,000 embeddings into memory without PCA
- PCA with 128 dimensions has 4x faster search times
- PCA with 128 dimensions is capable of doing GPU bitext retrieval on 
2 million lines in 17 minutes with 86% accuracy
- Clustering is inefficient for single search, but might be worth investigating for repeated search
- PCA and Clustering affect cosine similarity statistics, so disable --cutoff unaligned_score
- PCA and Clustering reduce accuracy
- LaBSE outputs 768 bit embeddings which may be slightly faster to search than 1024 bit LASER embeddings
- It may also be possible to run a multi-stage lookup generating a candidate list via
PCA lookup and then verifying cosine similarity against the original 768/1024 bit embedding.

It may also be possible to run a multi-stage lookup generating a candidate list via
PCA lookup and then verifying cosine similarity against the original 768/1024 bit embedding.


### Locality Sensitive Hashing
- https://en.wikipedia.org/wiki/Locality-sensitive_hashing
- https://coursera.org/share/b7f022d417576d5aceb2b4e751e0916a

Another possible (untested) usecase for Embeddings + PCA is Locality Sensitive Hashing.

Locality Sensitive Hashing creates a bitmask of N bytes by drawing random planes
over the vector space, with the bits representing on which side each vector is
relative to each planes. This is the implementation for K Nearest Neighbors.
FAISS may do something similar to this with its clustered indexing.

For cosine similarity search, this idea might be modified for angular coordinates
by doing PCA down to N dimensions and testing if
`cosine_similarity( PCA(embedding, N), eigenvector ) > 0`
for each of the eigenvectors, to generate an N bit hash.

These hashes could be uploaded to Elasticsearch and combined with identity search
for bucketing or index segmentation. The idea being to reduce the search space of
candidates to be searched for.


## Generate Embeddings
```
reports/faiss/generate_embeddings.py
```

## Links
- https://github.com/facebookresearch/faiss/
- https://github.com/kyamagu/faiss-wheels  (pypi: faiss-cpu + faiss-gpu)

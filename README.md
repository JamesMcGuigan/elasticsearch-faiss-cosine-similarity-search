# ElasticSearch Cosine Similarity Search
Code: [./reports/elasticsearch/](./reports/elasticsearch)

## GCP ElasticSearch
- Console: https://cloud.elastic.co/deployments/85c2199a9c984972bd29e8e14879912b
- Kibana:  https://16836df578ca49e89b4e0dfcb3993ca5.europe-west1.gcp.cloud.es.io:9243/app/home
- ElasticSearch: https://147b50e12302499b937da6d79c1d887a.europe-west1.gcp.cloud.es.io:9243

## Links

Tutorial:
- https://www.elastic.co/blog/text-similarity-search-with-vectors-in-elasticsearch

To Investigate:
- https://github.com/StaySense/fast-cosine-similarity
- https://docs.aws.amazon.com/elasticsearch-service/latest/developerguide/knn.html
    - https://opendistro.github.io/for-elasticsearch-docs/docs/knn/ - KNN requires AWS or OpenDistro
- https://github.com/facebookresearch/faiss - FAISS

ElasticSearch Guide:
- https://github.com/JamesMcGuigan/elasticsearch-tweets


## Install ElasticSearch Locally 

Ubuntu/Debian Install
- https://www.elastic.co/guide/en/elasticsearch/reference/current/deb.html

```
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
sudo apt-get install apt-transport-https
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-7.x.list
sudo apt-get update && sudo apt-get install elasticsearch

sudo vim /etc/elasticsearch/jvm.options        # optional: -Xms2g -Xmx2g
sudo vim /etc/elasticsearch/elasticsearch.yml  # cluster.routing.allocation.disk.threshold_enabled: false

sudo -i service elasticsearch stop
sudo -i service elasticsearch start
```

## Ingest
```
sudo npm install -g json5
reports/elasticsearch/schema.sh
reports/elasticsearch/ingest.sh
```

## Query
```
reports/elasticsearch/query.py -v -t "Madam President, on a point of order." -c 10
reports/elasticsearch/query.py -v -t "Four Score and Seven Years Ago"
reports/elasticsearch/query.py -v -t "I should like to observe a minute's silence" --lang italian 
```

```
reports/elasticsearch/query.py -v -t "I should like to observe a minute's silence" --lang Italian
query_embedding() timings: {'embedding': 0.867, 'query': 13.203}
0.850088        Italian         Vi chiedo di osservare un minuto di silenzio.
0.848983        Italian         Vi invito a osservare un minuto di silenzio.
0.843102        Italian         Vi invito pertanto a osservare un minuto di silenzio.
0.838669        Italian         Il Parlamento osserva un minuto di silenzio
0.835562        Italian         Vi chiedo ora di osservare un minuto di silenzio.
0.823920        Italian         Vi invito a osservare questo minuto di silenzio.
0.820581        Italian         Onorevoli deputati, vi chiedo di osservare un minuto di silenzio.
0.819016        Italian         Onorevoli parlamentari, vi chiedo di osservare un minuto di silenzio.
0.818605        Italian         Onorevoli colleghi, propongo un minuto di silenzio.
0.817192        Italian         Vi invito, onorevoli colleghi, ad osservare un minuto di silenzio.
```

```
reports/elasticsearch/query.py -v -t "Madam President, on a point of order." -c 20
query_embedding() timings: {'embedding': 0.86, 'query': 34.034}
1.000000        English         Madam President, on a point of order.
0.990433        English         Mr President, on a point of order.
0.990433        English            Mr President, on a point of order.
0.989137        Spanish         Señora Presidenta, sobre una cuestión de orden.
0.983990        Portuguese      Senhora Presidente, é para um ponto de ordem.
0.983581        Spanish         Señora Presidenta, una cuestión de orden.
0.983429        Portuguese      Senhor Presidente, para um ponto de ordem.
0.982367        French          Madame la Présidente, sur une motion de procédure.
0.982063        Danish          Fru formand, en bemærkning til forretningsordenen.
0.981689        Spanish         Señor Presidente, sobre una cuestión de orden.
0.981465        German          Frau Präsidentin, eine Bemerkung zur Geschäftsordnung.
0.981197        Portuguese      Senhora Presidente, intervenho para um ponto de ordem.
0.980337        Italian         Signora Presidente, su una mozione di procedura.
0.979547        Dutch           Mevrouw de Voorzitter, een motie van orde.
0.979357        Portuguese      Senhor Presidente, um ponto de ordem.
0.978980        Portuguese      Senhor Presidente, relativamente a um ponto de ordem.
0.978890        Italian         Signora Presidente, per un richiamo al regolamento.
0.978755        German          Frau Präsidentin, ein Wort zur Geschäftsordnung.
0.978430        Spanish         Señor Presidente, para una cuestión de orden.
0.976959        German          Frau Präsidentin, zur Geschäftsordnung.
```

# FAISS

## Findings
[reports/faiss/faiss_demo.py](reports/faiss/faiss_demo.py)

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

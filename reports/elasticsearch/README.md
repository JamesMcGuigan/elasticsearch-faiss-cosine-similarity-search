# ElasticSearch Cosine Similarity Search

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


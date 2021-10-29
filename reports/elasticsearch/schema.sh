#!/usr/bin/env bash
# Non-destructive alterative: https://github.com/JamesMcGuigan/elasticsearch-tweets/blob/master/server/SchemaUpdate.mjs

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit
set -x

if [[ ! "$ELASTICSEARCH_URL" ]]; then
    source .env.gcp
fi

curl -w "\n" -X DELETE "$ELASTICSEARCH_URL/$INDEX"
curl -w "\n" -X PUT    "$ELASTICSEARCH_URL/$INDEX" --data "$(json5 $SCHEMA)" -H "Content-Type: application/json"
curl -w "\n" -X GET    "$ELASTICSEARCH_URL/_cat/indices"

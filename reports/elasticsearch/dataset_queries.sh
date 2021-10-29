#!/usr/bin/env bash
#
# This script queries ElasticSearch and caches results to the filesystem
# Mostly useful as we have a 14 day free trial and might want to analyze results after this date

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd ../../
source venv/bin/activate

set -x
INPUT_GLOB=./datasets/europarl/it-en/*
OUTPUT_DIR=./output/elasticsearch/europarl/

mkdir -p ./output/elasticsearch/europarl
find $INPUT_GLOB | xargs -L1 head -n 100 | sort | uniq | while read SENTENCE || [[ -n $SENTENCE ]];
do
    BASENAME=$(echo $SENTENCE | perl -p -e 's/[^\n\w]+/_/g')
    FILENAME="$OUTPUT_DIR/$BASENAME.txt"
    if [[ ! -f $FILENAME ]]; then
        (
            echo "\$ reports/elasticsearch/query.py -v -c 100 -t \"$SENTENCE\"";
            CUDA_VISIBLE_DEVICES="" reports/elasticsearch/query.py -v -c 100 -t "$SENTENCE"
        ) > $FILENAME
    fi
done

#!/bin/bash

BASE_PATH="."

mkdir -p $BASE_PATH/data_store/averitec
mkdir -p $BASE_PATH/knowledge_store

if [ ! -f "$BASE_PATH/data_store/averitec/dev.json" ]; then
    wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/dev.json -O $BASE_PATH/data_store/averitec/dev.json
fi

if [ ! -d "$BASE_PATH/knowledge_store/dev" ]; then
    wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/dev_knowledge_store.zip -O $BASE_PATH/knowledge_store/dev_knowledge_store.zip
    unzip $BASE_PATH/knowledge_store/dev_knowledge_store.zip -d $BASE_PATH/knowledge_store/
    mv $BASE_PATH/knowledge_store/output_dev $BASE_PATH/knowledge_store/dev
    rm $BASE_PATH/knowledge_store/dev_knowledge_store.zip
fi

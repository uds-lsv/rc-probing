#!/usr/bin/env bash

MODEL=$1
FIRST_LAYER=$2
LAST_LAYER=$3
SUBSET=$4
export CUDA_VISIBLE_DEVICES=$5


# training data
for layer in $(seq $FIRST_LAYER $LAST_LAYER);
do
    for pooler in 'cls' 'mean'
    do
        for split in 'train'
        do
            echo "Model: ${MODEL} -- Layer: ${layer} -- Pooler: ${pooler} -- Split: ${split}"
            python /vectorizer/main.py --config /vectorizer/vectorizer/configs/rc-probing-subsets/${SUBSET}/${MODEL}/${split}.yaml --layer ${layer} --pooler ${pooler} --cuda
        done
    done
done


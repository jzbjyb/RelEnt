#!/bin/bash

set -e

in_dir=$1
shuf_dir=$2
top_dir=$3
top=$4

for file in ${in_dir}/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        shuf_fn=$shuf_dir/$filename
        top_fn=$top_dir/$filename
        echo "$file -> $shuf_fn -> $top_fn"
        shuf $file > $shuf_fn
        head -n $top $shuf_fn > $top_fn
    fi
done
#!/bin/bash


if [ "$#" != 3 ]; then
    echo "usage: this_script.sh 100-random-seeds.txt <start_seed_number> <end_seed_number>"
    exit
fi

seed_list=`head -$3 100-random-seeds.txt | tail -$(($3-$2+1))`

pushd ../sm_cnn/

for seed in $seed_list;
do
    echo $seed
    /usr/bin/time -o $seed.time.log python main.py sm_cnn.seed=$seed.model --paper-ext-feats --random-seed $seed > sm_cnn.seed=$seed.log 2>&1 
done

popd

#!/bin/bash


if [ "$#" != 1 ]; then
    echo "usage: this_script.sh 100-random-seeds.txt" 
    echo "generates a files with columns: seed, time, MAP, MRR"
    exit
fi

seed_list=`cat 100-random-seeds.txt`

pushd ../sm_cnn/

for seed in $seed_list;
do
    echo $seed >> seeds.tmp
    cat $seed.time.log | grep elapsed | cut -f3 -d' ' | sed 's_elapsed__g' >> times.tmp
    cat ../sm_cnn/sm_cnn.seed=$seed.log | grep -w epoch | tail -1 | cut -f6 -d' '  >> epochs.tmp
    cat sm_cnn.seed=$seed.log | grep "\-raw-test" | cut -f5 -d' ' >> maps.tmp
    cat sm_cnn.seed=$seed.log | grep "\-raw-test" | cut -f8 -d' ' >> mrrs.tmp
done

paste seeds.tmp times.tmp epochs.tmp maps.tmp mrrs.tmp > ../variation-100-seeds/sm_cnn.results.txt
rm seeds.tmp times.tmp epochs.tmp maps.tmp mrrs.tmp

popd

tar czf sm_cnn.seeds.gold.submission.files.tar.gz ../../data/TrecQA/gold.seed* ../../data/TrecQA/submission.seed*


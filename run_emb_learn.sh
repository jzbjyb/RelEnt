#!/bin/bash
#SBATCH --mem=20000
#SBATCH --cpus-per-task=8
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out
set -e

text=$1
type=$2
use_tbow=$3
method=$4

dataset_dir=data_new/analogy_dataset/split_middle_dedup_nocommonchild_by_entail_nway_subgraph100_sample5_parent_occ_popu/
subprop_file=data_new/property_occurrence_prop435k_split_dedup/subprops_random
emb_file=../pytorch_big_graph/emb_new_dedup_shuf_default/transe.txt
#word_emb_file='data/emb/glove.6B.50d.txt'
#word_emb_size=50
#suffix='.tbow'

if [[ $type == 'sent' ]]; then
    sent_emb_file=${dataset_dir}/emb/${text}_${type}.txt
    sent_emb_size=50
    sent_suffix=.${text}_${type}
elif [[ $type == 'bow' ]]; then
    word_emb_file=${dataset_dir}/emb/${text}_${type}.txt
    word_emb_size=50
    suffix=.${text}_${type}
else
    word_emb_file='data/emb/glove.6B.50d.txt'
    word_emb_size=50
    suffix=.${text}_${type}
fi

word_emb_file2='data/emb/glove.6B.50d.txt'
word_emb_size2=50
suffix2='.tbow'

#for seed in 0 1 2 3 4 2019 1000 256 77 9541
for seed in 2019 1000 256 77 9541
do
    echo "======="
    echo ${seed} ${use_tbow}
    echo "======="

    if [[ $type == 'sent' ]]; then
        python run_emb_learn.py \
            --dataset_dir ${dataset_dir} \
            --subprop_file ${subprop_file} \
            --emb_file ${emb_file} \
            --use_sent ${use_tbow} \
            --sent_emb_size ${sent_emb_size} \
            --sent_emb_file ${sent_emb_file} \
            --sent_suffix ${sent_suffix} \
            --num_occs 100 \
            --lr 0.001 \
            --epoch 500 \
            --early_stop 100 \
            --seed ${seed} \
            --only_one_sample_per_prop \
            --filter_labels \
            --sent_emb_method ${method} \
            #--no_cuda
            #--save ranks
    elif [[ $type == 'all' ]]; then
        python run_emb_learn.py \
            --dataset_dir ${dataset_dir} \
            --subprop_file ${subprop_file} \
            --emb_file ${emb_file} \
            --word_emb_size ${word_emb_size} \
            --use_tbow ${use_tbow} \
            --suffix ${suffix} \
            --num_occs 100 \
            --lr 0.001 \
            --epoch 500 \
            --early_stop 100 \
            --seed ${seed} \
            --only_one_sample_per_prop \
            --filter_labels \
            --no_cuda
            #--save ranks
    else
        python run_emb_learn.py \
            --dataset_dir ${dataset_dir} \
            --subprop_file ${subprop_file} \
            --emb_file ${emb_file} \
            --word_emb_file ${word_emb_file} \
            --word_emb_size ${word_emb_size} \
            --use_tbow ${use_tbow} \
            --suffix ${suffix} \
            --num_occs 100 \
            --lr 0.001 \
            --epoch 500 \
            --early_stop 100 \
            --seed ${seed} \
            --only_one_sample_per_prop \
            --filter_labels \
            --no_cuda
            #--save ranks
    fi
done

: '
--use_sent ${use_tbow} \
--sent_emb_size ${sent_emb_size} \
--sent_emb_file ${sent_emb_file} \
--sent_suffix ${sent_suffix} \
'

: '
--word_emb_file ${word_emb_file} \
--word_emb_size ${word_emb_size} \
--use_tbow ${use_tbow} \
--suffix ${suffix} \
'

# ./run_emb_learn.sh dep bow 10 0
# ./run_emb_learn.sh dep sent 5 rnn_last
# ./run_emb_learn.sh dep sent 5 cnn_mean
# ./run_emb_learn.sh dep sent 5 avg
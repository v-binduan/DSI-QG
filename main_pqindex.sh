#!/bin/bash
cd pq_index/

K=256
D=6
input_emb_path=/mnt/blob/v-binduan/NQ/Datasets/new_datasets/downloads/data/retriever/triviaqa_nci/Doc_Index_PQ_6_256_seq196/triviaqa_bert_base_emb_196.tsv
task_name=initial_index
emb_size=768
output_path=/mnt/blob/v-binduan/NQ/Datasets/new_datasets/downloads/data/retriever/triviaqa_nci/Doc_Index_PQ_6_256_seq196/

python ./tools/graphindex_stable_sdc_init_single.py 1 $K $D $input_emb_path $task_name $emb_size $output_path

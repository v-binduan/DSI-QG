#!/bin/bash
cd pq_tools/

K=256
D=6
#input_emb_path=/mnt/blob/v-binduan/NQ/Datasets/nq_preprocess/Doc_Index/doc_content_embedding_bert_32.tsv
input_emb_path=/mnt/blob/v-binduan/NQ/Datasets/nq_preprocess/doc_content_embedding_t5_epoch_2.tsv
task_name=initial_index
emb_size=768
output_path=/mnt/blob/v-binduan/NQ/Datasets/nq_preprocess/Doc_Index_6_256_t5/

python ./tools/graphindex_stable_sdc_init_single.py 1 $K $D $input_emb_path $task_name $emb_size $output_path


K=$1
D=$2
input_emb_path=$3
task_name=$4
emb_size=$5
output_path=$6

echo "python ./tools/graphindex_stable_sdc_init_single.py 1 $K $D $input_emb_path $task_name $emb_size"
python ./tools/graphindex_stable_sdc_init_single.py 1 $K $D $input_emb_path $task_name $emb_size
if [ $? -ne 0 ]; then

    echo "failed"
    exit 1

else

    echo "succeed"

fi
cp -r init_index $output_path


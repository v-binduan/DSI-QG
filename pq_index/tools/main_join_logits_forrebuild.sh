#!/bin/bash

#--------------------------------
#Author: liujie34@baidu.com
#Date: 2020-03-02 21:08:48
#--------------------------------

epoch=$1
date=$2

AFS_CLIENT=/root/paddlejob/hadoop-client/hadoop/bin/hadoop
PYTHON_CLIENT_p=/app/ecom/targeting/yeshaogui/tdm_qt/fluid_python2_1.8.tar.gz

input_query_logits=/app/ecom/targeting/yeshaogui/tdm_qt/${date}/query_logits_gdm/${epoch}
input_query_ad=/app/ecom/targeting/yeshaogui/tdm_qt/${date}/01_qt_pair_click
output=/app/ecom/targeting/yeshaogui/tdm_qt/${date}/query_logits_gdm/${epoch}_rebuild_ad_logits

map_cmd="fluid_python2_1.8/bin/python join_logits.py map_qad"
red_q_cmd="fluid_python2_1.8/bin/python join_logits.py red_join_q_logits 8 20"
red_ad_cmd="fluid_python2_1.8/bin/python join_logits.py red_joinad 8 20"


job_name="[Research][AUCTION][gdm_qt_tdm_eval_${date}][yeshaogui]"
job_name="fz_fc_research_liujie34_QT_GDM_${date}_gen_ad_path_scores"


hadoop_conf_file="hadoop-site.xml"
hadoop_conf_file="hadoop-site_xibu.xml"

echo "start process"
function process_func()
{
    ${AFS_CLIENT} fs -conf ${hadoop_conf_file} -test -e ${output}
    if [ $? -eq 0 ];then
        ${AFS_CLIENT} fs -conf ${hadoop_conf_file} -rmr ${output}
    fi

    ${AFS_CLIENT} streaming \
        -conf ${hadoop_conf_file} \
        -D abaci.is.dag.job=true \
        -D mapred.job.name=${job_name} \
        -D mapred.job.map.capacity=1024 \
        -D mapred.job.reduce.capacity=10000 \
        -D mapred.map.tasks=1024 \
        -D mapred.reduce.tasks=1000 \
        -D mapred.job.priority=VERY_HIGH \
        -D stream.memory.limit=4000 \
        -D abaci.dag.vertex.num=4 \
        -D abaci.dag.next.vertex.list.0=2 \
        -D abaci.dag.next.vertex.list.1=2 \
        -D abaci.dag.next.vertex.list.2=3 \
        -D mapred.input.dir.0=${input_query_ad} \
        -D mapred.input.dir.1=${input_query_logits} \
        -D stream.map.streamprocessor.0="${map_cmd}" \
        -D stream.reduce.streamprocessor.2="${red_q_cmd}" \
        -D stream.reduce.streamprocessor.3="${red_ad_cmd}" \
        -input  ${input_query_ad} \
        -output ${output} \
        -mapper "cat" \
        -reducer "cat" \
        -file    "join_logits.py" \
        -cacheArchive ${PYTHON_CLIENT_p}
}
process_func
echo "process done"

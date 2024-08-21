#!/bin/bash

rpc_limit=$1
config=$2
server=$3
IO500_Dir="/mnt/IOLustre/io500"
YEAR=$4
MONTH=$5
DAY=$6
concurrent_runs=$7
samples=100
WAIT_VAL=1
run_pid=$$
echo $$ > chaos_script_pid_$server.txt


sample_array=()
if [ ${concurrent_runs} -gt 0 ]; then
    for run in $(seq 1 ${samples}); do

        if [ ! -d "/root/darshan-analysis/applications/IO500/darshan-traces/${rpc_limit}_rpc_${server}_chaos" ]; then
            mkdir /root/darshan-analysis/applications/IO500/darshan-traces/${rpc_limit}_rpc_${server}_chaos
        fi

        echo "Creating chaos run ${run} on ${server}"
        rm -f pid_chaos_${server}.txt

        ssh -f root@${server} "cd ${IO500_Dir}; rm -f pid_${server}.txt; \
        nohup ./read_ior_hard.sh > /dev/null 2>&1 & echo \$! > pid_${server}.txt; \
        while [ ! -s pid_${server}.txt ]; do sleep 1; done; \
        scp pid_${server}.txt root@10.18.195.149:/root/darshan-analysis/applications/IO500/scripts/pid_chaos_${server}.txt; \
        wait \$(cat pid_${server}.txt);"
        # nohup ./io500_chaos.sh ${config} ${server}_${run_pid}_${run} > /dev/null 2>&1 & echo \$! > pid_${server}.txt;

        while [ ! -s pid_chaos_${server}.txt ]; do sleep 1; done
        pid=$(cat pid_chaos_${server}.txt)
        sample_array+=(${pid})
        if [ ${run} -lt ${concurrent_runs} ]; then
            sleep ${WAIT_VAL}
        else
            replace=0 
            while [ ${replace} -eq 0 ]; do
                for pid in ${sample_array[@]}; do
                    if [ $(ssh -f root@${server} "ps --pid ${pid} | wc -l") -eq 2 ]; then
                        echo "${pid} is running"
                    else
                        echo "${pid} is not running"
                        #remove pid from array
                        sample_array=(${sample_array[@]/$pid})
                        replace=$((replace+1))
                        # ssh -f root@${server} "cd ${IO500_Dir}/results; scp -r $(find . -maxdepth 1 -type d -name "*${run_pid}_${run}" -print) root@10.18.195.149:darshan-analysis/applications/IO500/darshan-traces/${rpc_limit}_rpc_${server}_chaos"
                        break
                    fi
                done
                sleep 1
            done
        fi
    done
fi

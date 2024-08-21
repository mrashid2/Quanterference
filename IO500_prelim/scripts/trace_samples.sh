#/usr/bin/env bash

rpc_limit=$1
config=$2
samples=$3
SERVERS=$4
IO500_Dir="/mnt/IOLustre/io500"
concurrent_runs=1
run_pid=$$
rm -rf metering_run.txt
echo $$ > metering_run.txt
echo trace samples pid: $$ > multi_config_pid.txt

YEAR=$5
MONTH=$6
DAY=$7

MODEL_SERVER=$8

for server in ${SERVERS}; do
    echo "Running all benchmarks ${samples} times on ${server}}"

    if [ ! -d "/root/darshan-analysis/applications/IO500/darshan-traces/${rpc_limit}_rpc_${server}_new" ]; then
        mkdir /root/darshan-analysis/applications/IO500/darshan-traces/${rpc_limit}_rpc_${server}_new
    fi

    sample_array=()
    current_runs=()
    if [ ${concurrent_runs} -gt 0 ]; then
        for run in $(seq 1 ${samples}); do
            start_time=$(date +%Y-%m-%dT%H:%M:%S)
            echo "Creating metering run ${run} on ${server} at ${start_time}" >> multi_config.log

            echo "Running ${run} concurrent jobs on ${server}"
            rm -f pid.txt

            ssh -f root@${server} "cd ${IO500_Dir}; rm -f pid_${server}.txt; \
            nohup ./io500.sh ${config} /darshan-logs/${YEAR}/${MONTH}/${DAY}/IO500_rpc_${rpc_limit}_sample_${run}.darshan ${run_pid}_${run} > /dev/null 2>&1 & echo \$! > pid_${server}.txt; \
            while [ ! -s pid_${server}.txt ]; do sleep 1; done; \
            scp pid_${server}.txt root@10.18.195.149:/root/darshan-analysis/applications/IO500/scripts/pid.txt; \
            wait \$(cat pid_${server}.txt); \
            scp /darshan-logs/${YEAR}/${MONTH}/${DAY}/IO500_rpc_${rpc_limit}_sample_${run}.darshan root@10.18.195.149:darshan-analysis/applications/IO500/darshan-traces/${rpc_limit}_rpc_${server}_new; \
            rm -rf /darshan-logs/${YEAR}/${MONTH}/${DAY}/IO500_rpc_${rpc_limit}_sample_${run}.darshan"

            # ssh to model server and run python script in background and save pid to file
            #ssh -f root@${MODEL_SERVER} "cd /root/Darshan-analyzer/python-files/stream_intercept; \
            #rm -f pid.txt; \
            #nohup python3 live_model.py > /dev/null 2>&1 & echo \$! > pid.txt;"


            while [ ! -s pid.txt ]; do sleep 1; done
            pid=$(cat pid.txt)
            echo "pid is ${pid}" >> multi_config.log
            sample_array+=(${pid})
            current_runs+=(${run})
            if [ ${run} -lt ${concurrent_runs} ]; then
                sleep ${WAIT_VAL}
            else
                replace=0 
                while [ ${replace} -eq 0 ]; do
                    idx=0
                    for pid in ${sample_array[@]}; do
                        if [ $(ssh -f root@${server} "ps --pid ${pid} | wc -l") -eq 2 ]; then
                            echo "${pid} is running"
                        else
                            stop_time=$(date +%Y-%m-%dT%H:%M:%S)
                            echo "${pid} stopped running at ${stop_time}" >> multi_config.log
                            echo "wall time: $(($(date +%s -d ${stop_time}) - $(date +%s -d ${start_time})))" >> multi_config.log
                            echo "${pid} is not running"
                            #kill python script on model server
                            #ssh -f root@${MODEL_SERVER} "cd /root/Darshan-analyzer/python-files/stream_intercept; \
                            #kill -9 $(cat pid.txt); mv live_model_results.txt model_results/live_model_results_${stop_time}.txt; mv live_model_run.txt model_results/live_model_run_${stop_time}.txt; \
                            #rm -rf /mnt/IOLustre/darshan_stream.txt; rm -rf /mnt/IOLustre/server_stream/*;"
                            #remove pid from array
                            sample_array=(${sample_array[@]/$pid})
                            current_run=${current_runs[$idx]}
                            current_runs=(${current_runs[@]/$current_run})
                            replace=$((replace+1))
                            ssh -f root@${server} "cd ${IO500_Dir}/results; folder_name=$(find . -maxdepth 1 -type d -name \"*${run_pid}_${current_run}\" -print); real_folder=${folder_name%%${run_pid}_${current_run}}; scp -r $(find . -maxdepth 1 -type d -name \"$real_folder\" -print) root@10.18.195.149:darshan-analysis/applications/IO500/darshan-traces/${rpc_limit}_rpc_${server}_new"
                            break
                        fi
                        idx=$((idx+1))
                    done
                    sleep 1
                done
            fi
        done
    fi
done

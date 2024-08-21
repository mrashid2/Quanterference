#/usr/bin/env bash
interference_level=$1
multi_config_timestamp=$2
INTERFERENCE_CONFIG_LIST="None ior-easy-read.ini ior-easy-write.ini ior-hard-read.ini ior-hard-write.ini mdt-easy-write.ini mdt-hard-read.ini mdt-hard-write.ini"
cur_dir=$(pwd)
node="192.168.0.16"
truncate -s 0 multi_config.log
echo "Running all benchmarks on ${node}" >> multi_config.log

# check if the directory exists
if [ ! -d "/root/darshan-analysis/applications/IO500/results/multi_config_run_${multi_config_timestamp}" ]; then
    mkdir /root/darshan-analysis/applications/IO500/results/multi_config_run_${multi_config_timestamp}
fi
if [ ! -d "/root/darshan-analysis/applications/IO500/stats/multi_config_run_${multi_config_timestamp}" ]; then
    mkdir /root/darshan-analysis/applications/IO500/stats/multi_config_run_${multi_config_timestamp}
fi
if [ ! -d "/root/darshan-analysis/applications/IO500/darshan-traces/multi_config_run_${multi_config_timestamp}" ]; then
    mkdir /root/darshan-analysis/applications/IO500/darshan-traces/multi_config_run_${multi_config_timestamp}
fi

global_start=$(date +%Y-%m-%dT%H:%M:%S)
for config in ${INTERFERENCE_CONFIG_LIST}; do
    # echo "remounting lustre" >> multi_config.log
    # ./mount_IOLustre.sh
    # echo "finished remounting lustre" >> multi_config.log

    echo "beginning stat collection with interference level of ${interference_level}" >> multi_config.log
    stat_start=$(date +%Y-%m-%d_%H-%M)
    ssh root@${node} "iokit-gather-stats /bin/iokit-config start"
    echo "started stat collection at ${stat_start}" >> multi_config.log
    config_start_timestamp=$(date +%Y-%m-%dT%H:%M:%S)
    echo "Running ${config}" >> multi_config.log
    ./simulate_network.sh ${config} ${multi_config_timestamp} ${interference_level}
    echo "Finished running ${config}" >> multi_config.log
    config_end_timestamp=$(date +%Y-%m-%dT%H:%M:%S)
    echo "${config} start: ${config_start_timestamp}" >> multi_config.log
    echo "${config} end: ${config_end_timestamp}" >> multi_config.log
    echo "${config} wall time: $(($(date +%s -d "${config_end_timestamp}") - $(date +%s -d "${config_start_timestamp}")))" >> multi_config.log
    ssh root@${node} "cd /mnt/IOLustre/io500; mv results ${config}_${interference_level}_results;"
    timestamp=$(date +%Y-%m-%d_%H-%M)
    ssh root@${node} "iokit-gather-stats /bin/iokit-config stop ${config}_${timestamp}.tgz; scp ${config}_${timestamp}.tgz root@10.18.195.149:darshan-analysis/applications/IO500/stats/multi_config_run_${multi_config_timestamp}/${config}_${interference_level}.tgz; rm -rf ${config}_${timestamp}.tgz"
    stat_end=$(date +%Y-%m-%d_%H-%M)
    echo "finished stat collection at ${stat_end}" >> multi_config.log
done
global_end=$(date +%Y-%m-%dT%H:%M:%S)
echo "finished all configs at ${global_end}" >> multi_config.log
echo "global start: ${global_start}" >> multi_config.log
echo "global end: ${global_end}" >> multi_config.log
echo "total wall time: $(($(date +%s -d ${global_end}) - $(date +%s -d ${global_start})))" >> multi_config.log

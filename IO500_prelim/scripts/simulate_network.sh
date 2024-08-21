#/usr/bin/env bash

CHAOS_SERVERS="192.168.0.9 192.168.0.12 192.168.0.11 192.168.0.10 192.168.0.4 192.168.0.15 192.168.0.7"
MODELLING_SERVER="192.168.0.7"
METERING_SERVER="192.168.0.16"
IO500_Dir="/mnt/IOLustre/io500"
SAMPLES=3
config_full="config-custom.ini"
config_chaos=$1
YEAR=$(date +%Y)
MONTH=$(date +%-m)
DAY=$(date +%-d)
multi_config_timestamp=$2
interference_level=$3

truncate -s 0 sample_collection.log
echo "Running all benchmarks ${SAMPLES} times on ${CHAOS_SERVERS}" >> sample_collection.log
echo "Running all benchmarks ${SAMPLES} times on ${METERING_SERVER}" >> sample_collection.log
echo "Running config ${config_chaos}" >> sample_collection.log
echo "Running config ${config_full}" >> sample_collection.log

if [[ $config_chaos = "None" ]]; then
  echo "skipping chaos runs as config is 'None'"
else
  for server in ${CHAOS_SERVERS}; do
    ssh -f root@${server} "cd ${IO500_Dir}; rm -f ${config_chaos};"
    scp ${config_chaos} root@${server}:${IO500_Dir}
  done
fi
ssh -f ${METERING_SERVER} "cd ${IO500_Dir}; rm -f ${config_full};"
scp ${config_full} root@${METERING_SERVER}:${IO500_Dir}

CHAOS_PID=""
if [[ $config_chaos = "None" ]]; then
  echo "skipping chaos runs as config is 'None'"
else
  if [[ $interference_level -eq 0 ]]; then
    echo "skipping chaos runs as level is 0"
  else
    for server in ${CHAOS_SERVERS}; do
        rm -f chaos_script_pid_$server.txt
        ./start_chaos_runs.sh 10000 $config_chaos $server $YEAR $MONTH $DAY $interference_level &
        while [ ! -s chaos_script_pid_$server.txt ]; do sleep 1; done
        CHAOS_PID+="$(cat chaos_script_pid_$server.txt) "
    done
    sleep 60
  fi
fi



start_timestamp=$(date +%Y-%m-%dT%H:%M:%S)
if [[ $config_chaos = "None" ]]; then
  ./trace_samples.sh 10000 $config_full 3 $METERING_SERVER $YEAR $MONTH $DAY $MODELLING_SERVER
else
  ./trace_samples.sh 10000 $config_full $SAMPLES $METERING_SERVER $YEAR $MONTH $DAY $MODELLING_SERVER
fi

end_timestamp=$(date +%Y-%m-%dT%H:%M:%S)
echo "sample collection start: ${start_timestamp}" >> multi_config.log
echo "sample collection end: ${end_timestamp}" >> multi_config.log
echo "sample collection wall time: $(($(date +%s -d ${end_timestamp}) - $(date +%s -d ${start_timestamp})))" >> multi_config.log
kill -9 ${CHAOS_PID}

for server in ${METERING_SERVER}; do

  timestamp=$(date +%Y-%m-%d_%H-%M)
  for file in $(ls /root/darshan-analysis/applications/IO500/darshan-traces/10000_rpc_${server}_new); do
    echo "Moving ${file} to /root/darshan-analysis/applications/IO500/darshan-traces/${server}_${timestamp}_${config_chaos}_done/IO500_${10000}_${timestamp}.darshan"
    mkdir /root/darshan-analysis/applications/IO500/darshan-traces/multi_config_run_${multi_config_timestamp}/${server}_${timestamp}_${config_chaos}_${interference_level}_done
    mv /root/darshan-analysis/applications/IO500/darshan-traces/10000_rpc_${server}_new/${file} /root/darshan-analysis/applications/IO500/darshan-traces/multi_config_run_${multi_config_timestamp}/${server}_${timestamp}_${config_chaos}_${interference_level}_done/${file}
    
  done
  sleep 5
done

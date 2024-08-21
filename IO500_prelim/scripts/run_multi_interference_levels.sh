#/usr/bin/bash

INTERFERENCE_LEVELS=(3)
multi_config_run_timestamp=$(date +%Y-%m-%d_%H-%M)

for interference_level in ${INTERFERENCE_LEVELS[@]}; do
    ./run_multi_config.sh ${interference_level} ${multi_config_run_timestamp}
    run_timestamp=$(date +%Y-%m-%d_%H-%M)
    mv multi_config.log ./interference_logs/IO500_interference_run_${run_timestamp}.log
done
./unpack_stats.sh ${multi_config_run_timestamp}




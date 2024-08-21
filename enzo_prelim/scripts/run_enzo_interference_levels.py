import paramiko
import time
import datetime
import os
import subprocess
import datetime

logging_file = "./run_ramp_interference.log"

# Record the starting timestamp formatted as yyyy-mm-dd_hh-mm
interference_run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

# Create or truncate the log file and write the starting timestamp
with open(logging_file, "w") as f:
    f.write(f"Starting interference run at {interference_run_timestamp}\n")

# List of nodes to connect to
#"192.168.0.9" not working
nodes = [
    "192.168.0.12",
    "192.168.0.11",
    "192.168.0.10",
    "192.168.0.4",
    "192.168.0.15",
    "192.168.0.7",
]
main_node = "192.168.0.16"

# Function to create an SSH connection
def create_ssh_connection(ip, username="root", key_filepath=None):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if key_filepath:
        ssh.connect(ip, username=username, key_filename=key_filepath)
    else:
        # Password method (Not recommended)
        ssh.connect(ip, username=username)
    return ssh

def start_io500(node, count, chaos=False):
    try:
        now = datetime.datetime.now()
        year, month, day = now.year, now.month, now.day
        timestamp = now.strftime("%Y-%m-%d_%H-%M")
        if chaos:
            print(f"Starting chaos run on {node}")
            print(year, month, day)
            ssh = create_ssh_connection(node)
            stdin, stdout, stderr = ssh.exec_command(f"cd /mnt/IOLustre/io500; nohup ./run_ior.sh {node} {count} > /dev/null 2>&1 & echo $!")
            pid = stdout.readline().strip()
        else:
            ssh = create_ssh_connection(node)
            stdin, stdout, stderr = ssh.exec_command(f"cd /mnt/IOLustre/enzo-dev/run/Hydro/Hydro-3D/CollapseTestNonCosmological; nohup ./run_enzo.sh /darshan-logs/{year}/{month}/{day}/enzo.darshan > /dev/null 2>&1 & echo $!")
            pid = stdout.readline().strip()
        with open(logging_file, "a") as f:
            f.write(f"Started benchmark on {node} with PID {pid}\n")
    except Exception as e:
        with open(logging_file, "a") as f:
            f.write(f"Failed to start processes on the node {node}: {e}\n")
        print(f"Failed to start processes on the node {node}: {e}")
        exit(1)
    start_time = datetime.datetime.now()
    return ssh, pid, start_time


def is_process_running(pid):
    check_command = f"if ps -p {pid} > /dev/null; then echo 'running'; else echo 'not running'; fi"
    stdin, stdout, stderr = ssh_main.exec_command(check_command)
    output = stdout.readline().strip()
    return output == 'running'

interference_types = ['None', 'All-1', 'All-12']
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

for t in interference_types:
    if t == 'None':
        ssh_main, main_pid, start_time = start_io500(main_node, 1)
    else:
        level = int(t.split('-')[1])
        pids_per_node = {}
        for lev in range(1, level+1):
            for node in nodes:
                ssh, pid, start_time = start_io500(node, lev, chaos=True)
                if node not in pids_per_node:
                    pids_per_node[node] = []
                pids_per_node[node].append(pid)
        
        ssh_main, main_pid, start_time = start_io500(main_node, 1)
        
    done = False
    ending_time = None     
    while not done:
        # check if max rounds reached or process has exited
        if not is_process_running(main_pid):
            with open(logging_file, "a") as f:
                f.write(f"Main node has exited\n")
            ending_time = datetime.datetime.now()
            done = True
        time.sleep(1)
    wall_time = ending_time - start_time
    with open(logging_file, "a") as f:
        f.write(f"Main node has finished after {wall_time}\n")
        f.write(f"Main node has finished, killing all scripts on the other nodes\n")

    if t != 'None':
        for node in nodes:
            try:
                for pid in pids_per_node[node]:
                    ssh = create_ssh_connection(node)
                    ssh.exec_command(f"kill -9 {pid}")
                    ssh.close()
                ssh = create_ssh_connection(node)
                ssh.exec_command("kill -9 $(pgrep ior)")
                ssh.close()
                with open(logging_file, "a") as f:
                    f.write(f"Killed io500 on {node}\n")
            except Exception as e:
                print(f"Failed to kill io500 on {node}: {e}")

    try:

        if not os.path.exists(f"/root/darshan-analysis/applications/enzo/darshan-traces/multi_config_run_{interference_run_timestamp}"):
            os.makedirs(f"/root/darshan-analysis/applications/enzo/darshan-traces/multi_config_run_{interference_run_timestamp}")
        if not os.path.exists(f"/root/darshan-analysis/applications/enzo/darshan-traces/multi_config_run_{interference_run_timestamp}/192.168.0.16_done"):
            os.makedirs(f"/root/darshan-analysis/applications/enzo/darshan-traces/multi_config_run_{interference_run_timestamp}/192.168.0.16_done")
        now = datetime.datetime.now()
        year, month, day = now.year, now.month, now.day
        ssh_main.exec_command(f"scp /darshan-logs/{year}/{month}/{day}/*.darshan root@10.18.195.149:darshan-analysis/applications/enzo/darshan-traces/multi_config_run_{interference_run_timestamp}/192.168.0.16_done/enzo_{t}.darshan; rm -rf /darshan-logs/{year}/{month}/{day}/*.darshan")
        time.sleep(5)

        with open(logging_file, "a") as f:
            f.write(f"Darshan logs moved\n")
    except Exception as e:
        with open(logging_file, "a") as f:
            f.write(f"Failed to finalize stats collection: {e}\n")
        print(f"Failed to finalize stats collection: {e}")

        


        




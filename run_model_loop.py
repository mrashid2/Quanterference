import os
import subprocess


# get current working directory
cwd = os.getcwd()

models_dir = os.path.join(cwd, "NN_Model/models")
scripts_dir = os.path.join(cwd, "NN_Model/scripts")
scalers_dir = os.path.join(cwd, "NN_Model/scalers")
eval_results_dir = os.path.join(cwd, "eval_results")
if not os.path.exists(eval_results_dir):
    os.makedirs(eval_results_dir)

data_dir = os.path.join(cwd, "NN_Model/data")
workloads = [d for d in os.listdir(data_dir) if not d.startswith('.')]

# each dir in data_dir is a workload with train/test data
for workload in workloads:
    # train_test script expects train and test dirs as lists
    train_workloads = []
    test_workloads = []
    # check if 'train' and 'test' dirs exist
    if not os.path.exists(os.path.join(data_dir, workload, "train")) or not os.path.exists(os.path.join(data_dir, workload, "test")):
        # dir has workloads as subdirs 
        # only dlio is set up this way as it contains data from 
        # multiple deep learning models as separate workloads
        subdirs = [d for d in os.listdir(os.path.join(data_dir, workload)) if not d.startswith('.')]
        for subdir in subdirs:
            train_workloads.append(os.path.join(data_dir, workload, subdir))
            test_workloads.append(os.path.join(data_dir, workload, subdir))
    else:
        train_workloads.append(os.path.join(data_dir, workload))
        test_workloads.append(os.path.join(data_dir, workload))

    print(f"\n\nRUNNING TRAIN TEST FOR {workload}\n\n")
    # run train_test_model.py for each workload
    subprocess.run(["python3", os.path.join(scripts_dir, "train_test_model.py"), 
                    "--train_workloads", *train_workloads, 
                    "--test_workloads", *test_workloads,
                    "--model_path", models_dir,
                    "--scaler_path", scalers_dir,
                    "--results_path", eval_results_dir])




import os
import subprocess


def run_model_loop(cwd, eval_results_dir, num_bins=2):
    models_dir = os.path.join(cwd, "NN_Model/models")
    scripts_dir = os.path.join(cwd, "NN_Model/scripts")
    scalers_dir = os.path.join(cwd, "NN_Model/scalers")

    data_dir = os.path.join(cwd, "NN_Model/data")
    workloads = [d for d in os.listdir(data_dir) if not d.startswith('.')]

    for workload in workloads:
        if num_bins != 2 and workload != "io500":
            continue
        train_workloads = []
        test_workloads = []
        if not os.path.exists(os.path.join(data_dir, workload, "train")) or not os.path.exists(os.path.join(data_dir, workload, "test")):
            subdirs = [d for d in os.listdir(os.path.join(data_dir, workload)) if not d.startswith('.')]
            for subdir in subdirs:
                train_workloads.append(os.path.join(data_dir, workload, subdir))
                test_workloads.append(os.path.join(data_dir, workload, subdir))
        else:
            train_workloads.append(os.path.join(data_dir, workload))
            test_workloads.append(os.path.join(data_dir, workload))

        print(f"\n\nRUNNING TRAIN TEST FOR {workload}\n\n")

        subprocess.run(["python3", os.path.join(scripts_dir, "train_test_model.py"), 
                        "--train_workloads", *train_workloads, 
                        "--test_workloads", *test_workloads,
                        "--model_path", models_dir,
                        "--scaler_path", scalers_dir,
                        "--results_path", eval_results_dir,
                        "--output_bins", str(num_bins)])
        


def generate_enzo_prelim_plots(cwd, eval_results_dir):
    enzo_prelim_levels_data_dir = os.path.join(cwd, "enzo_prelim/data/levels_analysis")
    enzo_prelim_types_data_dir = os.path.join(cwd, "enzo_prelim/data/types_analysis")
    enzo_prelim_scripts_dir = os.path.join(cwd, "enzo_prelim/scripts")

    # run script to generate enzo under different interference levels plot
    subprocess.run(["python3", os.path.join(enzo_prelim_scripts_dir, "plot_enzo.py"),
                    "--enzo_path", enzo_prelim_levels_data_dir,
                    "--output_path", eval_results_dir,
                    "--analysis_type", "levels"])

    # run script to generate enzo under different interference types plot
    subprocess.run(["python3", os.path.join(enzo_prelim_scripts_dir, "plot_enzo.py"),
                    "--enzo_path", enzo_prelim_types_data_dir,
                    "--output_path", eval_results_dir,
                    "--analysis_type", "types"])



if __name__ == "__main__":

    # get current working directory
    cwd = os.getcwd()
    eval_results_dir = os.path.join(cwd, "eval_results")
    if not os.path.exists(eval_results_dir):
        os.makedirs(eval_results_dir)

    run_model_loop(cwd, eval_results_dir, num_bins=2)   
    run_model_loop(cwd, eval_results_dir, num_bins=3)
    generate_enzo_prelim_plots(cwd, eval_results_dir)





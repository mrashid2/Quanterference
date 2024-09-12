import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import argparse


def plot_enzo_analysis(enzo_data_path, output_path, analysis_type):

    # Dictionary to store traces
    traces = {}

    print(f'Parsing CSV files in {enzo_data_path}')

    # Iterate over all files in the directory
    for file in os.listdir(enzo_data_path):
        # find all subdirectories
        if os.path.isdir(os.path.join(enzo_data_path, file)):
            sub_folder_path = os.path.join(enzo_data_path, file)
            for sub_file in os.listdir(sub_folder_path):
                if sub_file.endswith(".csv"):
                    # Parse the CSV file
                    trace_df = pd.read_csv(os.path.join(sub_folder_path, sub_file))
                    key = sub_file.split('.')[0]
                    traces[key] = trace_df

    # Plot the time of each read/write operation for each file
    max_duration = 0
    for key in traces:
        trace = traces[key]
        # Filter for only the first 500 operations
        trace = trace[trace.index < 1000]
        traces[key] = trace
        if trace['duration'].max() > max_duration:
            max_duration = trace['duration'].max()

    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 5))

    base_color = 'green'
    high_color = 'red'
    mdt_color = 'blue'

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    window_size = 10

    trace = traces['No']
    
    # Smooth the duration data using moving average
    window_size = 10
    smoothed_duration = moving_average(trace['duration'], window_size)
    #smoothed_size = moving_average(trace['size'], window_size)
    
    sns.lineplot(x=trace.index, y=smoothed_duration, legend=False, label=f'No interference', color=base_color, alpha=0.4)
    
    plt.xlabel('Index of operation')
    plt.ylabel('Duration (ms)')
    plt.yscale('log')
    plt.ylim(0, max_duration + 500)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    for key in traces:
        if key == 'No':
            continue
        if key == 'Mid' or key == 'mid-High':
            continue
        if key == 'ior-hard' or key == 'ior-rnd4k' or key == 'mdt-hard':
            continue
        trace = traces[key]

        # Smooth the duration data using moving average
        smoothed_duration = moving_average(trace['duration'], window_size)
        
        # Plot the duration of each read/write operation
        if key == 'No':
            sns.lineplot(x=trace.index, y=smoothed_duration, legend=False, label=f'No interference', color=base_color, alpha=0.4)
        elif key == 'High' or key == 'ior-easy':
            sns.lineplot(x=trace.index, y=smoothed_duration, legend=False, label=f'{key} interference', color=high_color, alpha=0.7)
        elif key == 'mdt-easy':
            sns.lineplot(x=trace.index, y=smoothed_duration, legend=False, label=f'{key} interference', color=mdt_color, alpha=0.7)
        else:
            sns.lineplot(x=trace.index, y=smoothed_duration, legend=False, label=f'{key} interference', color="blue", alpha=0.7)

        plt.xlabel('Index of operation')
        plt.ylabel('Duration (ms)')
        plt.yscale('log')
        plt.ylim(0, max_duration + 500)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    if analysis_type == 'levels':
        output_file = os.path.join(output_path, f'enzo_prelim_interference_{analysis_type}[Fig1.a].png')
    else:
        output_file = os.path.join(output_path, f'enzo_prelim_interference_{analysis_type}[Fig1.b].png')
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Plot Enzo analysis results.')
    parser.add_argument('--enzo_path', type=str, required=True, help='Path to the Enzo data directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output plot')
    parser.add_argument('--analysis_type', type=str, choices=['levels', 'types'], required=True, help='Type of analysis to perform (read or write)')
    
    args = parser.parse_args()

    plot_enzo_analysis(args.enzo_path, args.output_path, args.analysis_type)

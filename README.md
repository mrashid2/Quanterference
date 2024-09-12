# Quanterference
## Generating evaluation results:
 - [generate_eval_results.py](./generate_eval_results.py) is a runnable script which does the following:
     - Trains/tests a model from scratch for each benchmark workload dataset resulting in recreation of the confusion matrices shown in Figure 3.
     - Trains/tests a model from scratch for each real application workload dataset resulting in recreation of the confusion matrices shown in Figure 5.
     - Trains/tests a 3-bin classification model on the io500 benchmark dataset resulting in the recreation of the confusion matrix shown in Figure 4.
     - Processes the enzo analysis trace data (from section II) to recreate plots a and b from Figure 1. 

Note: Each of the generated figures will be saved to the [eval_results](./eval_results/) directory inluding the Figure which it represents in brackets. ie. '...\[Fig5.a\].png' corresponds to the confusion matrix 'a' in figure 5.

## Repository Organization
The organization of the rest of this repository is the following:
 - [IO500_prelim](./IO500_prelim) contains the data and execution scripts used in our preliminary quantitative analysis of IO500 under various types of interference
      - [data](./IO500_prelim/data) contains the raw results used as a basis to calculate the relative slowdowns represented in Table 1
      - [scripts](./IO500_prelim/scripts) contains the series of scripts used to generate the above mentioned data. The process is launched by the `background_multi_interference.sh` script which launches the other scripts during its exectution
 - [enzo_prelim](./enzo_prelim) contains the data and execution scripts used in our prelimary quantitative analysis of Enzo under various types and levels of interference
      - [data](./enzo_prelim/data) contains the unprocessed (.Darshan) and processed (.csv) trace data from each exection reprsenented in Figure 1.
      - [scripts](./enzo_prelim/scripts) contains the scripts used to generate the above mentioned traces.
           - [run_enzo_interference_levels.py](./enzo_prelim/scripts/run_enzo_interference_levels.py) is the script used to run the enzo workload under various levels of cross-application interference.
           - [run_enzo_interference_types.py](./enzo_prelim/scripts/run_enzo_interference_types.py) is the script used to run the enzo workload under various different types of cross-application interference.
 - [NN_Model](./NN_Model) contains the training/testing data, training execution scripts, and model file for the trained model corresponding to Figures 3-5
      - [data](./NN_Model/data) contains the raw data collected from each application split into train and test sets for model training and evaluation
      - [model](./NN_Model/model) contains the trained model .pkl file
      - [scripts](./NN_Model/scripts) contains the model training and evaluation script


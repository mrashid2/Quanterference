# Quanterference

The organization of this repository is the following:
 - [IO500_prelim](./IO500_prelim) contains the data and execution scripts used in our preliminary quantitative analysis of IO500 under various types of interference
      - [data](./IO500_prelim/data) contains the raw results used as a basis to calculate the relative slowdowns represented in Table 1
      - [scripts](./IO500_prelim/scripts) contains the series of scripts used to generate the above mentioned data. The process is launched by the `background_multi_interference.sh` script which launches the other scripts during its exectution
 - [enzo_prelim](./enzo_prelim) contains the data and execution scripts used in our prelimary quantitative analysis of Enzo under verious types and levels of interference
      - [data](./enzo_prelim/data) contains the unprocessed (.Darshan) and processed (.csv) trace data from each exection reprsenented in Figure 1
      - [scripts](./enzo_prelim/scripts) contains the scripts used to generate the above mentioned traces
 - [NN_Model](./NN_Model) contains the training/testing data, training execution scripts, and model file for the trained model corresponding to Figures 3-5
      - [data](./NN_Model/data) contains the raw data collected from each application split into train and test sets for model training and evaluation
      - [model](./NN_Model/model) contains the trained model .pkl file
      - [scripts](./NN_Model/scripts) contains the model training and evaluation script
 - [run_model_loop.py](./run_model_loop.py) is a runnable script which recreates the confusion matrix results show in figures 3 and 5. The script trains a model on each workload and tests the trained model on the corresponding test set for each workload.
      - Note: While running the the script the confusion matrices for each workload will be saved to the [eval_results](./eval_results/) directory

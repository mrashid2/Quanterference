import argparse
import os
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_processor import MetricsDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import pickle


#cloudlab cluster
#OST_SERVERS = ['ost0', 'ost1', 'ost2', 'ost3', 'ost4', 'ost5', 'ost6', 'ost7', 'ost8', 'ost9', 'ost10']
OST_SERVERS = ['ost0', 'ost1', 'ost2', 'ost3', 'ost4', 'ost5']
MDT_SERVERS = ['mdt']
SERVER_COLUMNS = ['read_ios', 'read_merges', 'sectors_read', 'time_reading', 'write_ios', 'write_merges', 'sectors_written', 'time_writing', 'in_progress', 'io_time', 'weighted_io_time']
AGG_METRICS = ['mean', 'std', 'sum']
TRACE_KEYS = ["total_time", "window_size"]
OST_TRACE_KEYS = ["total_ops", "total_size", "total_reads", "total_writes", \
                    "total_read_size", "total_write_size", "IOPS", "read_IOPS", "write_IOPS", "throughput", \
                    "read_throughput", "write_throughput"]
MDT_TRACE_KEYS = ["total_ops", "total_stat", "total_open", "total_close", "total_IOPS", "total_stat_IOPS", \
                    "total_open_IOPS", "total_close_IOPS"]
MDT_STAT_COLUMNS = len(SERVER_COLUMNS) * len(AGG_METRICS)
OST_STAT_COLUMNS = len(SERVER_COLUMNS) * len(AGG_METRICS)
MDT_TRACE_COLUMNS = len(MDT_TRACE_KEYS)
OST_TRACE_COLUMNS = len(OST_TRACE_KEYS)




class SensitivityModel(nn.Module):
    def __init__(self, hidden_size=16, server_out_size = 8, output_size=1, server_emb_size=32):
        super(SensitivityModel, self).__init__()
        mdt_input_width = len(SERVER_COLUMNS) * len(AGG_METRICS) + len(MDT_TRACE_KEYS) + 1
        print('mdt_input_width: ', mdt_input_width)
        ost_input_width = len(SERVER_COLUMNS) * len(AGG_METRICS) + len(OST_TRACE_KEYS) + 1
        print('ost_input_width: ', ost_input_width)
        self.server_out_size = server_out_size
        self.mdt_fc = nn.Linear(mdt_input_width, server_emb_size)
        self.mdt_fc_hidden = nn.Linear(server_emb_size, hidden_size)
        self.mdt_fc_out = nn.Linear(hidden_size, server_out_size)
        
        self.ost_fc = nn.Linear(ost_input_width, server_emb_size)
        self.ost_fc_hidden = nn.Linear(server_emb_size, hidden_size)
        self.ost_fc_out = nn.Linear(hidden_size, server_out_size)
        self.fc_bridge = nn.Linear(len(MDT_SERVERS)*server_out_size + len(OST_SERVERS)*server_out_size, hidden_size)
        #self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def mdt_forward(self, mdt):
        mdt = mdt.view(-1, mdt.shape[-1])
        mdt = self.mdt_fc(mdt)
        mdt = self.relu(mdt)
        mdt = self.mdt_fc_hidden(mdt)
        mdt = self.relu(mdt)
        mdt = self.mdt_fc_out(mdt)
        mdt = self.relu(mdt)
        mdt = mdt.view(-1, len(MDT_SERVERS)*self.server_out_size)
        return mdt

    def ost_forward(self, ost):
        ost = ost.view(-1, ost.shape[-1])
        ost = self.ost_fc(ost)
        ost = self.relu(ost)
        ost = self.ost_fc_hidden(ost)
        ost = self.relu(ost)
        ost = self.ost_fc_out(ost)
        ost = self.relu(ost)
        ost = ost.view(-1, len(OST_SERVERS)*self.server_out_size)
        return ost

    def forward(self, mdt, ost):
        mdt = self.mdt_forward(mdt)
        ost = self.ost_forward(ost)
        x = torch.cat((mdt, ost), dim=1)
        x = self.fc_bridge(x)
        x = self.relu(x)
        #x = self.fc_hidden(x)
        #x = self.relu(x)
        x = self.fc_out(x)
        x = self.sigmoid(x)
        return x
"""  

class PerOSTModel(nn.Module):
    def __init__(self, hidden_size=16, output_size=1, server_emb_size=32):
        super(SensitivityModel, self).__init__()
        ost_input_width = len(SERVER_COLUMNS) * len(AGG_METRICS) + len(OST_TRACE_KEYS) + 1
        print('ost_input_width: ', ost_input_width)
        self.ost_fc = nn.Linear(ost_input_width, server_emb_size)
        self.ost_fc_hidden = nn.Linear(server_emb_size, hidden_size)
        self.ost_fc_out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()



    def forward(self, ost):
        ost = ost.view(-1, ost.shape[-1])
        ost = self.ost_fc(ost)
        ost = self.relu(ost)
        ost = self.ost_fc_hidden(ost)
        ost = self.relu(ost)
        ost = self.ost_fc_out(ost)
        ost = self.sigmoid(ost)
        return ost

    
class PerMDTModel(nn.Module):
    def __init__(self, hidden_size=16, output_size=1, server_emb_size=32):
        super(SensitivityModel, self).__init__()
        mdt_input_width = len(SERVER_COLUMNS) * len(AGG_METRICS) + len(MDT_TRACE_KEYS) + 1
        print('mdt_input_width: ', mdt_input_width)
        self.mdt_fc = nn.Linear(mdt_input_width, server_emb_size)
        self.mdt_fc_hidden = nn.Linear(server_emb_size, hidden_size)
        self.mdt_fc_out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, mdt):
        mdt = mdt.view(-1, mdt.shape[-1])
        mdt = self.mdt_fc(mdt)
        mdt = self.relu(mdt)
        mdt = self.mdt_fc_hidden(mdt)
        mdt = self.relu(mdt)
        mdt = self.mdt_fc_out(mdt)
        mdt = self.sigmoid(mdt)
        return mdt





class SimpleModel(nn.Module):
    def __init__(self, hidden_size=8, output_size=1, server_emb_size=16):
        super(SimpleModel, self).__init__()
        mdt_input_width = (len(SERVER_COLUMNS) * len(AGG_METRICS) + len(MDT_TRACE_KEYS)) * len(MDT_SERVERS) 
        ost_input_width = (len(SERVER_COLUMNS) * len(AGG_METRICS) + len(OST_TRACE_KEYS)) * len(OST_SERVERS)
        self.fc1 = nn.Linear(mdt_input_width+ost_input_width+1, server_emb_size)
        self.fc2 = nn.Linear(server_emb_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, mdt, ost):
        # flatten mdt and ost
        mdt = mdt.view(-1, mdt.size(1) * mdt.size(2))
        mdt = mdt[:, :-1]
        ost = ost[:, :, :-1]
        last_val = mdt[:, -1]
        ost = ost.reshape(-1, ost.size(1) * ost.size(2))
        x = torch.cat((mdt, ost), 1)
        x = torch.cat((x, last_val.view(-1, 1)), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc_out(x)
        x = self.sigmoid(x)
        return x

"""

def train_model(model, train_loader, valid_loader, epochs=100000, lr=0.0005):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    valid_losses = []
    train_f1 = []
    valid_f1 = []
    best_model = model
    loss_steps = 5000
    train_loss = 0
    idx = 0
    for epoch in range(epochs):
        model.train()
        train_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        valid_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        for mdt_window, ost_window, labels in train_loader:
            mdt_window = mdt_window.float()
            ost_window = ost_window.float()
            labels = labels.float()
            optimizer.zero_grad()
            output = model(mdt_window, ost_window)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            train_loss += loss_val
            for i in range(len(output)):
                if output[i] > 0.5:
                    if labels[i] == 1:
                        train_metrics['tp'] += 1
                    else:
                        train_metrics['fp'] += 1
                else:
                    if labels[i] == 0:
                        train_metrics['tn'] += 1
                    else:
                        train_metrics['fn'] += 1
            idx += 1

            if idx % loss_steps == 0:
                train_losses.append(train_loss)
                train_loss = 0

                #valid_loss = 0
                #best_valid_loss = float('inf')
                #valid_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
                """
                with torch.no_grad():
                    model.eval()
                    for mdt_window, ost_window, labels in valid_loader:
                        mdt_window = mdt_window.float()
                        ost_window = ost_window.float()
                        labels = labels.float()
                        output = model(mdt_window, ost_window)
                        loss = criterion(output, labels)
                        valid_loss += loss.item()
                        for i in range(len(output)):
                            if output[i] > 0.5:
                                if labels[i] == 1:
                                    valid_metrics['tp'] += 1
                                else:
                                    valid_metrics['fp'] += 1
                            else:
                                if labels[i] == 0:
                                    valid_metrics['tn'] += 1
                                else:
                                    valid_metrics['fn'] += 1
                    valid_losses.append(valid_loss)
                """
        
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            for mdt_window, ost_window, labels in valid_loader:
                mdt_window = mdt_window.float()
                ost_window = ost_window.float()
                labels = labels.float()
                output = model(mdt_window, ost_window)
                loss = criterion(output, labels)
                valid_loss += loss.item()
                for i in range(len(output)):
                    if output[i] > 0.5:
                        if labels[i] == 1:
                            valid_metrics['tp'] += 1
                        else:
                            valid_metrics['fp'] += 1
                    else:
                        if labels[i] == 0:
                            valid_metrics['tn'] += 1
                        else:
                            valid_metrics['fn'] += 1
            valid_losses.append(valid_loss)

        print(f'Epoch {epoch} Train Loss: {train_loss}')
        print(f'TP: {train_metrics["tp"]} FP: {train_metrics["fp"]} TN: {train_metrics["tn"]} FN: {train_metrics["fn"]}')
        try:
            precision = train_metrics['tp'] / (train_metrics['tp'] + train_metrics['fp'])
        except:
            precision = 0
        try:
            recall = train_metrics['tp'] / (train_metrics['tp'] + train_metrics['fn'])
        except:
            recall = 0
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except:
            f1 = 0
        train_f1.append(f1)
        print(f'TRAIN: Precision: {precision} Recall: {recall} F1: {f1}')
        try:
            precision = valid_metrics['tp'] / (valid_metrics['tp'] + valid_metrics['fp'])
        except:
            precision = 0
        try:
            recall = valid_metrics['tp'] / (valid_metrics['tp'] + valid_metrics['fn'])
        except:
            recall = 0
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except:
            f1 = 0
        valid_f1.append(f1)
        print(f'TP: {valid_metrics["tp"]} FP: {valid_metrics["fp"]} TN: {valid_metrics["tn"]} FN: {valid_metrics["fn"]}')
        print(f'VALID: Precision: {precision} Recall: {recall} F1: {f1}')
    best_model = model
    return best_model, train_losses, valid_losses, train_f1, valid_f1

def test_model(model, test_datasets, before_fine_tune, fine_tune_steps=0):
    for test_loader, name in test_datasets:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            model.eval()
            for mdt_window, ost_window, labels in test_loader:
                mdt_window = mdt_window.float()
                ost_window = ost_window.float()
                labels = labels.float()
                output = model(mdt_window, ost_window)
                for i in range(len(output)):
                    all_labels.append(labels[i])
                    if output[i] > 0.5:
                        
                        all_preds.append(1)
                        if labels[i] == 1:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        all_preds.append(0)
                        if labels[i] == 0:
                            tn += 1
                        else:
                            fn += 1
        try:
            precision = tp / (tp + fp)
        except:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except:
            recall = 0
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except:
            f1 = 0
        print(f'{name} Results')
        print(f'\t\tTP: {tp} FP: {fp} TN: {tn} FN: {fn}')
        print(f'\t\tPrecision: {precision} Recall: {recall} F1: {f1}')

        cm = confusion_matrix(all_labels, all_preds)
        # save confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['< 2', '>= 2'])
        # save confusion matrix as png
        disp.plot(cmap='Blues')

        plt.savefig(f'confusion_matrix_{name}_{"before" if before_fine_tune else "after"}_fine_tune_{fine_tune_steps}.png')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--train_workloads', nargs='+', default=['/Users/chris/Downloads/darshan-analysis/python-files/data_files/io500'])
    parser.add_argument('--train_set_proportion', type=float, default=1.0)
    parser.add_argument('--augment', action='store_true', default=True)
    #parser.add_argument('--test_workloads', nargs='+', default=['/Users/chris/Downloads/darshan-analysis/python-files/data_files/h5bench-amrex', '/Users/chris/Downloads/darshan-analysis/python-files/data_files/h5bench-openpmd', '/Users/chris/Downloads/darshan-analysis/python-files/data_files/enzo', '/Users/chris/Downloads/darshan-analysis/python-files/data_files/io500', '/Users/chris/Downloads/darshan-analysis/python-files/data_files/dlio_bench_unet3d', '/Users/chris/Downloads/darshan-analysis/python-files/data_files/dlio_bench_bert'])
    parser.add_argument('--test_workloads', nargs='+', default=['/Users/chris/Downloads/darshan-analysis/python-files/data_files/enzo'])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--fine_tune', action='store_true', default=True)
    parser.add_argument('--fine_tune_workloads', nargs='+', default=['/Users/chris/Downloads/darshan-analysis/python-files/data_files/dlio_bench_unet3d', '/Users/chris/Downloads/darshan-analysis/python-files/data_files/dlio_bench_bert', '/Users/chris/Downloads/darshan-analysis/python-files/data_files/h5bench-amrex', '/Users/chris/Downloads/darshan-analysis/python-files/data_files/h5bench-openpmd', '/Users/chris/Downloads/darshan-analysis/python-files/data_files/enzo'])
    parser.add_argument('--fine_tune_epochs', type=int, default=30)
    parser.add_argument('--bin_thresholds', nargs='+', type=int, default=[2])
    args = parser.parse_args()

    real_workloads = ['/Users/chris/Downloads/darshan-analysis/python-files/data_files/h5bench-amrex', 
                      '/Users/chris/Downloads/darshan-analysis/python-files/data_files/h5bench-openpmd', 
                      '/Users/chris/Downloads/darshan-analysis/python-files/data_files/enzo']

    if args.train:
        train_dataset = MetricsDataset(args.train_workloads, True, args.bin_thresholds, augment=args.augment)
        scaler = train_dataset.scaler
        #train_dataset = MetricsDataset(args.train_workloads, True, args.bin_thresholds, augment=args.augment, scaler=scaler)
        if args.train_set_proportion < 1.0:
            train_size = int(args.train_set_proportion * len(train_dataset))
            train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset)-train_size])
            print('training set size: ', len(train_dataset))
        
    else:
        scaler = 'scaler'  

    test_datasets = []
    valid_datasets = []
    for workload in args.test_workloads:     
        test_dataset = MetricsDataset([workload], False, args.bin_thresholds, scaler)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        test_size = int(0.8 * len(test_dataset))
        print('test set size: ', len(test_dataset))
        print('postive samples: ', np.sum(test_dataset.target))
        print('negative samples: ', len(test_dataset) - np.sum(test_dataset.target))

        valid_size = len(test_dataset) - test_size
        valid_dataset, _ = torch.utils.data.random_split(test_dataset, [valid_size, test_size])
        valid_datasets.append(valid_dataset)
        
        print('validation set size: ', len(valid_dataset))
        test_datasets.append((test_loader, workload.split('/')[-1]))
    
    # concat all valid datasets
    valid_dataset = torch.utils.data.ConcatDataset(valid_datasets)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    

    if args.train:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        print('training set size: ', len(train_dataset))
        if args.train_set_proportion < 1.0:
            print('training set proportion: ', args.train_set_proportion)
        else:
            print('postive samples: ', np.sum(train_dataset.target))
            print('negative samples: ', len(train_dataset) - np.sum(train_dataset.target))



    if args.train:
        model = SensitivityModel()
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total number of parameters: {total_params}')
        model, train_losses, valid_losses, train_f1, valid_f1 = train_model(model, train_loader, valid_loader, epochs=args.epochs)
        fig = plt.figure()
        steps = np.array(range(len(train_losses))) * 5000
        train_losses = np.array(train_losses) / 5000
        valid_losses = np.array(valid_losses) / len(valid_loader)
        plt.plot(steps, train_losses, label='Training Loss')
        #plt.plot(steps, valid_losses, label='Validation Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Avg. Loss Per Sample')
        plt.legend(loc='upper right')
        plt.savefig('loss.png')
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
    else:
        print('loading model')
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total number of parameters: {total_params}')

    print('testing model')
    test_model(model, test_datasets, before_fine_tune=True)

    """
    if args.fine_tune:
        fine_tune_steps = 0
        train_workloads = args.train_workloads
        for workload in args.fine_tune_workloads:
            
            # sample data from all train workloads to add to fine tune training set
            train_samples = []
            for train_workload in train_workloads:
                train_dataset = MetricsDataset([train_workload], True, args.bin_thresholds, scaler, augment=args.augment)
                train_size = int(0.1 * len(train_dataset))
                train_set, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset)-train_size])
                train_samples.append(train_set)
            print('Previous training set samples: ', len(train_samples))
            fine_tune_train_dataset = MetricsDataset([workload], True, args.bin_thresholds, scaler, augment=args.augment)
            print('fine tune training set size: ', len(fine_tune_train_dataset))
            fine_tune_train_dataset = torch.utils.data.ConcatDataset([fine_tune_train_dataset] + train_samples)
            
            fine_tune_train_loader = DataLoader(fine_tune_train_dataset, batch_size=1, shuffle=True)
            print('fine tune training set size: ', len(fine_tune_train_dataset))


            fine_tune_test_dataset = MetricsDataset([workload], False, args.bin_thresholds, scaler)
            fine_tune_test_loader = DataLoader(fine_tune_test_dataset, batch_size=1, shuffle=True)
            fine_tune_test_size = int(0.8 * len(fine_tune_test_dataset))
            print('fine tune test set size: ', len(fine_tune_test_dataset))
            print('postive samples: ', np.sum(fine_tune_test_dataset.target))
            print('negative samples: ', len(fine_tune_test_dataset) - np.sum(fine_tune_test_dataset.target))


            fine_tune_valid_size = len(fine_tune_test_dataset) - fine_tune_test_size
            fine_tune_valid_dataset, _ = torch.utils.data.random_split(fine_tune_test_dataset, [fine_tune_valid_size, fine_tune_test_size])
            fine_tune_valid_loader = DataLoader(fine_tune_valid_dataset, batch_size=1, shuffle=True)
            print('fine tune validation set size: ', len(fine_tune_valid_dataset))
            

            model, train_losses, valid_losses, train_f1, valid_f1 = train_model(model, fine_tune_train_loader, fine_tune_test_loader, epochs=args.fine_tune_epochs)
            test_model(model, test_datasets, before_fine_tune=False, fine_tune_steps=fine_tune_steps)
            fine_tune_steps += 1
            train_workloads.append(workload)
    """



        







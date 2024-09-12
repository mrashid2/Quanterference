import argparse
import numpy as np
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
        if output_size == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = nn.Softmax(dim=1)

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
        x = self.last_activation(x)
        return x
    

def train_model(model, train_loader, valid_loader, epochs=100000, lr=0.0005, num_bins=2):
    if num_bins == 2:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    valid_losses = []
    train_f1 = []
    valid_f1 = []
    loss_steps = 5000
    train_loss = 0
    idx = 0
    for epoch in range(epochs):
        print(f'Currently running Epoch {epoch}')
        model.train()
        train_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        valid_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        for mdt_window, ost_window, labels in train_loader:
            mdt_window = mdt_window.float()
            ost_window = ost_window.float()
            if num_bins == 2:
                labels = labels.float()
            else:
                labels = labels.argmax(dim=1)  # Convert one-hot to class indices
            optimizer.zero_grad()
            output = model(mdt_window, ost_window)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            train_loss += loss_val
            if num_bins == 2:
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

        
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            for mdt_window, ost_window, labels in valid_loader:
                mdt_window = mdt_window.float()
                ost_window = ost_window.float()
                if num_bins == 2:
                    labels = labels.float()
                else:
                    labels = labels.argmax(dim=1)  # Convert one-hot to class indices
                output = model(mdt_window, ost_window)
                loss = criterion(output, labels)
                valid_loss += loss.item()
                valid_losses.append(valid_loss)
                if num_bins == 2:
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
            

        if num_bins == 2:
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
            print(f'VALID: Precision: {precision} Recall: {recall} F1: {f1}\n')
    return model, train_losses, valid_losses, train_f1, valid_f1

def test_model(model, test_datasets, results_path='../results/', num_bins=2):
    cm = None
    for test_loader, name in test_datasets:
        all_labels = []
        all_preds = []
        with torch.no_grad():
            model.eval()
            for mdt_window, ost_window, labels in test_loader:
                mdt_window = mdt_window.float()
                ost_window = ost_window.float()
                if num_bins == 2:
                    labels = labels.float()
                else:
                    labels = labels.argmax(dim=1)  # Convert one-hot to class indices
                output = model(mdt_window, ost_window)
                for i in range(len(output)):
                    if num_bins == 2:
                        all_labels.append(int(labels[i]))
                        pred = 1 if output[i] > 0.5 else 0
                        all_preds.append(pred)
                    else:  # num_bins == 3
                        all_labels.append(int(labels[i]))
                        all_preds.append(torch.argmax(output[i]))


        if num_bins == 2:
            tp = sum((p == 1 and l == 1) for p, l in zip(all_preds, all_labels))
            fp = sum((p == 1 and l == 0) for p, l in zip(all_preds, all_labels))
            tn = sum((p == 0 and l == 0) for p, l in zip(all_preds, all_labels))
            fn = sum((p == 0 and l == 1) for p, l in zip(all_preds, all_labels))

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
            print(f'{name} TestResults')
            print(f'\t\tTP: {tp} FP: {fp} TN: {tn} FN: {fn}')
            print(f'\t\tPrecision: {precision} Recall: {recall} F1: {f1}')

        new_cm = confusion_matrix(all_labels, all_preds)
        if cm is None:
            cm = new_cm
        else:
            cm += new_cm

    # save confusion matrix
    if num_bins == 2:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['< 2', '>= 2'])
    else:  # num_bins == 3
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['< 2', '[2, 5)', '>= 5'])
    
    disp.plot(cmap='Blues')

    if num_bins == 3:
        plt.savefig(f'{results_path}/confusion_matrix_io500_3bins_[Fig4].png')
    elif 'dlio' in name:
        plt.savefig(f'{results_path}/confusion_matrix_dlio_[Fig3.b].png')
    elif 'io500' in name:
        plt.savefig(f'{results_path}/confusion_matrix_io500_[Fig3.a].png')
    elif 'amrex' in name:
        plt.savefig(f'{results_path}/confusion_matrix_amrex_[Fig5.a].png')
    elif 'enzo' in name:
        plt.savefig(f'{results_path}/confusion_matrix_enzo_[Fig5.b].png')
    elif 'openpmd' in name:
        plt.savefig(f'{results_path}/confusion_matrix_openpmd_[Fig5.c].png')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--train_workloads', nargs='+', default=['/Users/chris/Downloads/darshan-analysis/python-files/data_files/io500'])
    parser.add_argument('--train_set_proportion', type=float, default=1.0)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--scaler_path', type=str, default='../scalers/scaler')
    parser.add_argument('--model_path', type=str, default='../model/model.pkl')
    parser.add_argument('--results_path', type=str, default='../results/')
    parser.add_argument('--test_workloads', nargs='+', default=['/Users/chris/Downloads/darshan-analysis/python-files/data_files/io500'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output_bins', type=int, default=2)
    args = parser.parse_args()

    if args.output_bins == 2:
        bin_thresholds = [2]
    elif args.output_bins == 3:
        bin_thresholds = [2, 5]

    if args.train:
        train_dataset = MetricsDataset(args.train_workloads, True, bin_thresholds, augment=args.augment, scaler_path=args.scaler_path)
        scaler = train_dataset.scaler
        if args.train_set_proportion < 1.0:
            train_size = int(args.train_set_proportion * len(train_dataset))
            train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset)-train_size])
        
    else:
        scaler = args.scaler_path

    test_datasets = []
    valid_datasets = []
    for workload in args.test_workloads:     
        test_dataset = MetricsDataset([workload], False, bin_thresholds, scaler)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        test_size = int(0.8 * len(test_dataset))

        valid_size = len(test_dataset) - test_size
        valid_dataset, _ = torch.utils.data.random_split(test_dataset, [valid_size, test_size])
        valid_datasets.append(valid_dataset)
        
        test_datasets.append((test_loader, workload.split('/')[-1]))
    
    # concat all valid datasets
    valid_dataset = torch.utils.data.ConcatDataset(valid_datasets)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    

    if args.train:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)



    if args.train:
        if args.output_bins == 2:
            model = SensitivityModel(output_size=1)
        else:
            model = SensitivityModel(output_size=args.output_bins)
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total number of parameters: {total_params}')
        model, train_losses, valid_losses, train_f1, valid_f1 = train_model(model, train_loader, valid_loader, epochs=args.epochs, num_bins=args.output_bins)
        fig = plt.figure()
        steps = np.array(range(len(train_losses))) * 5000
        train_losses = np.array(train_losses) / 5000
        valid_losses = np.array(valid_losses) / len(valid_loader)
        plt.plot(steps, train_losses, label='Training Loss')
        #plt.plot(steps, valid_losses, label='Validation Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Avg. Loss Per Sample')
        plt.legend(loc='upper right')
        if len(args.test_workloads) > 1:
            test_workload = ""
            test_workload = args.test_workloads[0].split('/')[-1]
            for workload in args.test_workloads[1:]:
                test_workload += "_" + workload.split('/')[-1]
        else:
            test_workload = args.test_workloads[0].split('/')[-1]

        with open(f"{args.model_path}_{args.output_bins}bins_model.pkl", 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(f"{args.model_path}_{args.output_bins}bins_model.pkl", 'rb') as f:
            model = pickle.load(f)
        total_params = sum(p.numel() for p in model.parameters())

    test_model(model, test_datasets, results_path=args.results_path, num_bins=args.output_bins)



        








import numpy as np
import sklearn.metrics
from numpy import r_, around
import torch
from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader

from data_processing.data_loader import ImageGraphDataset, minibatch_graphs
from .networks import init_graph_net
from . import evaluation
from data_processing.graph_io import project_nodes_to_img
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


BATCH_SIZE = 8

'''
#Input#
model_type is a string that determines the type of graph learning layers used (GraphSAGE, GAT)
hyperparameters is a named tuple defined in utils/hyperparam helpers
train_dataset is an ImageGraphDataset with read_graph set to True.
'''


class GNN:
    def __init__(self, hyperparameters, train_dataset):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device", self.device)
        print(torch.cuda.get_device_name(self.device))
        class_weights = torch.FloatTensor(
            hyperparameters.class_weights).to(self.device)
        self.net = init_graph_net(hyperparameters)
        # # load pre-trained model
        # path = '/home/mpatel74/projects/def-akilan/mpatel74/TRY500/GAT/try4/vah_try1.pt'
        # self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(
        ), lr=hyperparameters.lr, weight_decay=hyperparameters.w_decay)
        self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, hyperparameters.lr_decay, last_epoch=-1, verbose=False)
        self.loss_fcn = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                       collate_fn=minibatch_graphs) if train_dataset is not None else None

    def run_epoch(self):
        # print("3")
        self.net.train()
        # print("5")
        losses = []
        f1 = []
        # i = 0
        for _, batch_graph, batch_feats, batch_label in self.train_loader:
            # print("4")
            batch_graph = batch_graph.to(self.device)
            batch_feats = batch_feats.to(self.device)
            batch_label = batch_label.to(self.device)
            # batch_graph = batch_graph.to(self.device)
            # batch_labels = batch_label.to(self.device)
            logits = self.net(batch_graph, batch_feats)

            loss = self.loss_fcn(logits, batch_label)
            losses.append(loss.item())

            _, pred_classes = torch.max(logits, dim=1)
            pred_classes = pred_classes.detach().cpu().numpy()
            labels = batch_label.detach().cpu().numpy()
            f1_score = sklearn.metrics.f1_score(
                pred_classes, labels, average='micro')
            f1.append(f1_score.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print("Batch no:", i, "loss:", losses[-1], "acc:", acc[-1])
            # i += 1
        self.lr_decay.step()

        return np.mean(losses), np.mean(f1)

    # must be a Subset of an ImageGraphDataset
    def evaluate(self, dataset: ImageGraphDataset):
        assert(dataset.read_label == True)
        self.net.eval()
        # metrics stores loss,label counts, node dices,voxel dices,voxel hausdorff
        losses = []
        f1 = []
        for curr_ids, curr_graphs, curr_feat, curr_label in dataset:
            curr_graphs = curr_graphs.to(self.device)
            curr_feat = torch.FloatTensor(curr_feat).to(self.device)
            curr_label = torch.LongTensor(curr_label).to(self.device)
            with torch.no_grad():
                logits = self.net(curr_graphs, curr_feat)
                loss = self.loss_fcn(logits, curr_label)
                losses.append(loss.item())
            _, predicted_classes = torch.max(logits, dim=1)
            predicted_classes = predicted_classes.detach().cpu().numpy()
            curr_label = curr_label.detach().cpu().numpy()
            f1_score = sklearn.metrics.f1_score(
                predicted_classes, curr_label, average='micro')
            f1.append(f1_score.item())
        # print("Testing F1-score is", f1)
        return np.mean(losses), np.mean(f1)

    def save_weights(self, folder, name):
        torch.save(self.net.state_dict(), f"{folder}{name}.pt")

import torch
import os
import glob
import networkx as nx
import numpy as np
from dgl import from_networkx as to_dgl_graph
from dgl import batch as dgl_batch

# from torch_geometric.data import Batch
# # from torch_geometric.data.batch import from_data_list
# from torch_geometric.utils.convert import from_networkx
# from torch_geometric.loader import DataLoader


from data_processing import graph_io


class ImageGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_root_dir, all_ids, read_graph=True, read_label=True):
        self.data_root_dir = data_root_dir
        self.all_ids = all_ids
        self.read_graph = read_graph
        self.read_label = read_label

    def get_one(self, img_id):
        if (self.read_graph):
            return (img_id, *self.get_graph(img_id))
        else:
            print("Invalid combination of flags")

    def get_graph(self, img_id):

        # Load a NetworkX graph from a JSON file and extract node features and labels if applicable
        
        nx_graph = graph_io.load_networkx_graph(
            f"{self.data_root_dir}{os.sep}{img_id}{os.sep}{img_id}_nxgraph.json")
        feats = np.array([nx_graph.nodes[n]['features']
                          for n in nx_graph.nodes])
       
        # If reading the label is enabled, extract node labels as well
        if (self.read_label):
            labels = np.array([nx_graph.nodes[n]['label']
                               for n in nx_graph.nodes])
        
        # Convert the NetworkX graph to a DGL graph
        G = to_dgl_graph(nx_graph)
        n_edges = G.number_of_edges()
       
        # Compute the in-degree of each node and normalize it by taking the inverse square root
        deg = G.in_degrees().float()
        normaliz = torch.pow(deg, -0.5)
        normaliz[torch.isinf(normaliz)] = 0
        G.ndata['norm'] = normaliz.unsqueeze(1)

        # If reading the label is enabled, add node labels to the DGL graph
        # Return the DGL graph, node features, and node labels (if applicable)
        if (self.read_label):
            # G.ndata['label'] = labels
            # print("hello")
            # print(G, feats, labels)
            return G, feats, labels
        return G, feats

    def get_supervoxel_partitioning(self, img_id):
        fi = f"{self.data_root_dir}{os.sep}{img_id}{os.sep}{img_id}_region.npy"
        return np.load(fi)

    def __iter__(self):
        for img_id in self.all_ids:
            yield self.get_one(img_id)

    def __getitem__(self, index):
        img_id = self.all_ids[index]
        return self.get_one(img_id)

    def __len__(self):
        return len(self.all_ids)


def minibatch_graphs(samples):
    # print(samples)
    img_ids, graphs, feats, labels = map(list, zip(*samples))
#    batch_graph = Batch.from_data_list(graphs)
    # print("hello")
    batch_graph = dgl_batch(graphs)
    # exit()
    # print(batch_graph)
    return img_ids, batch_graph, torch.FloatTensor(np.concatenate(feats)), torch.LongTensor(np.concatenate(labels))


def collate_refinement_net(samples):
    net, Img, Lab = samples[0]
    return net, torch.FloatTensor(Img), torch.LongTensor(Lab)

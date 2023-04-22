from prettytable import PrettyTable
import argparse
from numpy import r_, around
from torch.utils.data import Subset
import torch
from utils.training_helpers import *
from utils.hyperparam_helpers import populate_hardcoded_hyperparameters
from data_processing.data_loader import ImageGraphDataset
from model.gnn_model import GNN
from model.networks import init_graph_net
from torchvision import models
#from torch_geometric.nn import summary
from utils.run_details import get_run_details
import os

#Set the device to cuda if it is available, otherwise set it to cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Load hardcoded hyperparameters for the graph neural network
gnn_hp = populate_hardcoded_hyperparameters()
#Initialize the graph neural network with the hyperparameters
net = init_graph_net(gnn_hp)
#Define the path to save the parameters of the graph neural network
fp = f"{get_run_details().run_name}{os.sep}{get_run_details().run_name}_parameters.txt"


from prettytable import PrettyTable

# Define a function to count the number of trainable parameters in a neural network
def count_parameters(net):
    # Create a table to display the information about each parameter
    table = PrettyTable(["Modules", "Para Tensor Shape", "Parameters"])
    total_params = 0 # Initialize the total number of parameters to zero
    
    # Iterate through each parameter in the network
    for name, parameter in net.named_parameters():
        # If the parameter doesn't require gradients, skip it
        if not parameter.requires_grad:
            continue
        # Get the shape and number of elements in the parameter tensor
        param_shape = list(parameter.size())
        params = parameter.numel()
        # Add a row to the table for the parameter
        table.add_row([name, param_shape, params])
        # Add the number of parameters to the total count
        total_params += params
    
    # Add a row to the table for the total number of parameters
    table.add_row(["Total Para:", "", total_params])
    print(table) # Print the table to the console
    
    # Print the network summary to the console
    print("\n")
    print(net)
    
    # Write the table to a file
    with open(fp, "w") as f:
        f.write(table.get_string())

count_parameters(net)

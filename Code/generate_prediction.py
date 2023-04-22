
import torch
import numpy as np
import random
import os
import glob
from model.gnn_model import GNN
from data_processing import graph_io, data_loader
from data_processing.data_loader import ImageGraphDataset
import matplotlib.pyplot as plt
from utils.run_details import get_run_details
from data_processing.image_processing import *
from utils.training_helpers import *
from skimage import io


from utils.hyperparam_helpers import populate_hardcoded_hyperparameters
from model.networks import init_graph_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Label_map = {0: (255, 255, 255),
             1: (0, 0, 255),
             2: (0, 255, 255),
             3: (0, 255, 0),
             4: (255, 255, 0),
             5: (255, 0, 0)}

Inverted_label_map = {color: no for no, color in Label_map.items()}

""" Numeric labels to RGB-color encoding """


def color_encoding(enc_label):
    dec_label = np.zeros(
        (enc_label.shape[0], enc_label.shape[1], 3), dtype=np.uint8)

    for no, color in Label_map.items():
        indices = enc_label == no
        dec_label[indices] = color

    return dec_label


''' RGB-color encoding to numeric labels'''


def numeric_encoding(dec_label):
    enc_label = np.zeros(
        (dec_label.shape[0], dec_label.shape[1]), dtype=np.uint8)

    for no, color in Label_map.items():
        indices = np.all(dec_label == np.array(color).reshape(1, 1, 3), axis=2)
        enc_label[indices] = no

    return enc_label


'''
Loads model weights from given file and generates predictions on graphs in specified data directory.
Data directory expected to created by preprocess script.
Outputs are saved in image form NOT in graph form. Output can either be logits (needed to train CNN) or predictions (for evaluation/downstream use)
Need to make sure that the type and shape of the weight file correspond to the model that the weights are being loaded into. See load_net_and_weights.
'''

# Make sure model type and hyperparameters correspond to the weight file.


def load_net_and_weights(weight_file):

    gnn_hp = populate_hardcoded_hyperparameters()
    net = init_graph_net(gnn_hp)

    net.load_state_dict(torch.load(weight_file, map_location=device))
    net.eval()
    return net


# both save formats save as image.
# if preds is selected as output format, then they will additionally be expanded back to original BraTS size (240,240,155)
def save_predictions(net, dataset):
    global device
    net = net.to(device)
    for mri_id, graph, feats in dataset:
        graph = graph.to(device)
        feats = torch.FloatTensor(feats).to(device)
        with torch.no_grad():
            logits = net(graph, feats)
        save_voxel_preds(mri_id, dataset, logits)


# Define a function to save voxel predictions as an image
def save_voxel_preds(img_id, Dataset, node_logit):
    global data_dir # Use the global variable 'data_dir'
    
    # Get the predicted node by finding the index of the maximum logit value
    _, predicted_node = torch.max(node_logit, dim=1)
    predicted_node = predicted_node.detach().cpu().numpy()
    
    # Get the supervoxel partitioning for the current image
    supervoxel_partitioning = Dataset.get_supervoxel_partitioning(img_id)
    
    # Project the predicted nodes onto the image and get the resulting voxel predictions
    predicted_voxel = graph_io.project_nodes_to_img(supervoxel_partitioning, predicted_node)
    
    # Convert the predicted voxel to an RGB image using color encoding
    predicted_voxel = color_encoding(predicted_voxel[:, :, 0])
    
    # Save the predicted voxel as an image with the file name "<img_id>_predicted.jpg"
    io.imsave(f"{data_dir}{os.sep}{img_id}{os.sep}{img_id}_predicted.jpg", predicted_voxel, check_contrast=False)


# Define a function to get the IDs of all images in a dataset
def get_all_imgs_in_dataset(data_root_dir):
    # Use the 'glob' module to find all subdirectories (i.e., image folders) within the specified root directory
    img_folders = glob.glob(f"{data_root_dir}**/*", recursive=True)
    
    # Extract the image IDs from the folder paths
    img_ids = [fp.split(os.sep)[-1] for fp in img_folders]
    
    # Print the number of images found
    print(f"Found {len(img_folders)} images")
    
    # Return the list of image IDs
    return img_ids



if __name__ == '__main__':
   # Get the details of the current run (not shown)
    dir_details = get_run_details()

    # Set the directories for input data and output files
    data_dir = f"{dir_details.run_name}{os.sep}processed"
    output_dir = f"{dir_details.run_name}"

    # Get the IDs of all images in the input data directory
    all_ids = get_all_imgs_in_dataset(os.path.expanduser(data_dir))

    # Shuffle the image IDs and split them into training, validation, and testing sets
    random.seed(100)
    random.shuffle(all_ids)
    total_imgs = len(all_ids)
    train_ids = all_ids[:int(0.6*total_imgs)]
    val_ids = all_ids[int(0.6*total_imgs):int(0.8*total_imgs)]
    test_ids = all_ids[int(0.8*total_imgs):]

    # Create dataset objects for the training, validation, and testing sets
    train_dataset = ImageGraphDataset(os.path.expanduser(
        data_dir), train_ids, read_graph=True, read_label=False)
    val_dataset = ImageGraphDataset(os.path.expanduser(
        data_dir), val_ids, read_graph=True, read_label=False)
    test_dataset = ImageGraphDataset(os.path.expanduser(
        data_dir), test_ids, read_graph=True, read_label=False)


    # output_dir = os.path.expanduser(output_dir)
    # if not os.path.isdir(output_dir):
    #     print(f"Creating save directory: {output_dir}")
    #     os.makedirs(output_dir)
    # dataset = data_loader.ImageGraphDataset(os.path.expanduser(
    #     args.data_dir), args.data_prefix, read_img=False, read_graph=True, read_label=False)

    #hyperparams = populate_hardcoded_hyperparameters(args.model_type)
    # Get the path to the saved weights file for the current run (not shown)
    weight_file = f"{dir_details.run_name}{os.sep}{dir_details.run_name}.pt"

    # Load the neural network and its weights from the saved file
    net = load_net_and_weights(os.path.expanduser(weight_file))

    # Generate and save predictions for the training, validation, and testing datasets using the loaded network
    save_predictions(net, train_dataset)
    save_predictions(net, val_dataset)
    save_predictions(net, test_dataset)


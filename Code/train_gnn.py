import argparse
from numpy import r_, around
from torch.utils.data import Subset
import os
import glob
import random
import time

from utils.training_helpers import *
from utils.hyperparam_helpers import populate_hardcoded_hyperparameters
from data_processing.data_loader import ImageGraphDataset
from model.gnn_model import GNN
from data_processing import graph_io
from data_processing.image_processing import *
from img2graph.graphgen import img2graph
from utils.run_details import get_run_details


# This function trains a GNN model on a full dataset and evaluates it on a test set
def train_on_full_dataset(dir_details, hyperparams, Dataset):
    # Print statement to indicate that training is starting
    print("Training on full dataset")
    
    # Initialize a GNN model with the specified hyperparameters and the training dataset
    model = GNN(hyperparams, Dataset["train_dataset"])
    
    # Record the current time to later determine how long training takes
    t = time.time()
    
    # Train the model on the training dataset for the specified number of epochs, saving the results in the specified directory
    train(model, dir_details.run_name+os.sep, hyperparams.n_epochs,
          dir_details.run_name, Dataset["val_dataset"])
    
    # Print the time it took to train the model
    print("Time taken for training :", time.time()-t)
    
    # Evaluate the model on the test dataset and retrieve the F1-score
    _, test_f1 = model.evaluate(Dataset["test_dataset"])
    
    # Print the F1-score
    print("Testing F1-score is", test_f1)



def get_all_imgs_in_dataset(data_root_dir):
    # Get all image file paths in the given directory and its subdirectories
    img_folders = glob.glob(f"{data_root_dir}**/*", recursive=True)
    
    # Extract image IDs from file paths and remove file extensions
    img_ids = [fp.split(os.sep)[-1][:-4] for fp in img_folders]
    
    # Print the number of images found
    print(f"Found {len(img_folders)} images")
    
    # Return the list of image IDs
    return img_ids



if __name__ == '__main__':

    # Get details of the current run
    dir_details = get_run_details()

    # Get the hardcoded hyperparameters for the model
    hyper_params = populate_hardcoded_hyperparameters()

    # Define directories for data input, output, and processing
    data_dir = f"{dir_details.run_name}{os.sep}processed"
    img_dir = dir_details.data_dir
    output_dir = f"{dir_details.run_name}"

    # Define the list of all image ids
    all_images_no = ['2_10', '2_11', '2_12', '2_13', '2_14', 
                '3_10', '3_11', '3_12', '3_13', '3_14', 
                '4_10', '4_11', '4_12', '4_13', '4_14', '4_15',
                '5_10', '5_11', '5_12', '5_13', '5_14', '5_15',
                '6_7', '6_8', '6_9', '6_10', '6_11', '6_12', '6_13', '6_14', '6_15',
                '7_7', '7_8', '7_9', '7_10', '7_11', '7_12', '7_13']

    # Get a list of all image ids present in the dataset
    all_ids = get_all_imgs_in_dataset(os.path.expanduser(img_dir))
    all_ids.sort()

    # Randomly shuffle the list of all image ids
    random.seed(100)
    random.shuffle(all_images_no)

    # Split the shuffled list of image ids into train, validation, and test sets
    test_no = all_images_no[:14]
    train_no = all_images_no[14:14+20]
    val_no = all_images_no[14+20:]
    train_ids = []
    val_ids = []
    test_ids = []

    # Assign each image id to the corresponding train, validation, or test set
    for img_id in all_ids:
        if img_id[12:15] in test_no or img_id[12:16] in test_no:
            test_ids.append(img_id)
        elif img_id[12:15] in train_no or img_id[12:16] in train_no:
            train_ids.append(img_id)
        else:
            val_ids.append(img_id)

    # Expand the train set by creating additional copies of each image with augmented data
    new_train_ids = []
    for id in train_ids:
        for i in range(hyper_params.aug_no):
            new_train_ids.append(f"{id}_{i}")

    # Create the train, validation, and test datasets
    train_dataset = ImageGraphDataset(os.path.expanduser(
        data_dir), new_train_ids, 1, read_graph=True, read_label=True)
    val_dataset = ImageGraphDataset(os.path.expanduser(
        data_dir), val_ids, 0, read_graph=True, read_label=True)
    test_dataset = ImageGraphDataset(os.path.expanduser(
        data_dir), test_ids, 0, read_graph=True, read_label=True)

    # Package the datasets into a dictionary for easy access
    Dataset = {"train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset}

    # Train the model on the full dataset
    train_on_full_dataset(dir_details, hyper_params, Dataset)

import numpy as np
import glob
import os
import concurrent.futures
import argparse
import cv2
from skimage import io
from skimage.segmentation import mark_boundaries
import time
import random

from data_processing import graph_io
from data_processing.image_processing import *
from img2graph.graphgen import img2graph
from utils.hyperparam_helpers import populate_hardcoded_hyperparameters
from utils.run_details import get_run_details
from utils.training_helpers import *
import torchvision.transforms as transforms

# Define a series of image transformations using PyTorch's transforms.Compose() function
DataAug_Transformation = transforms.Compose([
    
    # Convert the input image to a PIL (Python Imaging Library) image object
    transforms.ToPILImage(),
    
    # Randomly flip the image horizontally with a probability of 0.5 (commented out)
    # Randomly flip the image vertically with a probability of 0.5 (commented out)
    
    # Apply color jittering to the image, changing the brightness, contrast, and saturation with a factor of 0.5,
    # and the hue with a random value between -0.1 and 0.1.
    transforms.ColorJitter(0.5, 0.5, 0.5,(-0.1, 0.1))
])

#Set the number of threads to be used by the program to 50
N_THREADS = 50
#Set the STATS variable to None or a list of two lists of floating point numbers
STATS = None
# STATS = [[0.33862684, 0.36275441, 0.33696908],
#          [0.10296915, 0.10322405, 0.10652531]]

#Define a dictionary that maps integer labels to RGB colors
#The keys represent the integer label and the values are tuples of RGB values
#This is used for encoding the image labels

Label_map = {0: (255, 255, 255),
             1: (0, 0, 255),
             2: (0, 255, 255),
             3: (0, 255, 0),
             4: (255, 255, 0),
             5: (255, 0, 0)}

#Define an inverted dictionary that maps RGB colors to integer labels
#The keys represent tuples of RGB values and the values are the integer labels
#This is used for decoding the image labels

Inverted_label_map = {color: no for no, color in Label_map.items()}


# Define a function called "color_encoding" that takes in an "enc_label" argument
def color_encoding(enc_label):
    # Create a numpy array filled with zeros, with the same dimensions as "enc_label", but with a third dimension of size 3
    # This third dimension represents the RGB values of each pixel
    dec_label = np.zeros((enc_label.shape[0], enc_label.shape[1], 3), dtype=np.uint8)

    # Loop over each key-value pair in the "Label_map" dictionary
    for no, color in Label_map.items():
        # Find all the indices in "enc_label" that have a value equal to the current key (i.e., "no")
        indices = enc_label == no
        # Set the corresponding pixels in "dec_label" to the RGB values associated with the current key (i.e., "color")
        dec_label[indices] = color

    # Return the resulting "dec_label" array
    return dec_label



import numpy as np

# Define a function called "numeric_encoding" that takes in a "dec_label" argument
def numeric_encoding(dec_label):
    # Create a numpy array filled with zeros, with the same dimensions as "dec_label"
    enc_label = np.zeros((dec_label.shape[0], dec_label.shape[1]), dtype=np.uint8)

    # Loop over each key-value pair in the "Label_map" dictionary
    for no, color in Label_map.items():
        # Find all the indices in "dec_label" that have an RGB value equal to the current color
        # This is done by comparing each pixel's RGB value with the color value, reshaped to the same dimensions as "dec_label"
        # The result is a boolean array where True corresponds to pixels that match the current color
        indices = np.all(dec_label == np.array(color).reshape(1, 1, 3), axis=2)
        
        # Set the corresponding pixels in "enc_label" to the current label value (i.e., "no")
        enc_label[indices] = no

    # Return the resulting "enc_label" array
    return enc_label



class DataPreprocessor():
# Define a class called "SomeClassName" with an __init__ method that takes in "dir_details" and "hyper_params" arguments
    def __init__(self, dir_details, hyper_params):
        # Print a message to indicate that the code is preparing to build graphs
        print("Preparing to build graphs...")

        # Graph specifications:
        # Set the number of nodes in the graph to the "n_nodes" parameter from "hyper_params"
        self.num_node = hyper_params.n_nodes
        # Set the "boxiness_coef" to the "boxiness" parameter from "hyper_params"
        self.boxiness_coef = hyper_params.boxiness
        # Set the number of augmentations to the "aug_no" parameter from "hyper_params"
        self.aug_no = hyper_params.aug_no

        # Data specifications:
        # Set the data directory to "dir_details.data_dir"
        self.data_Dir = dir_details.data_dir
        # Set the output directory to "dir_details.run_name"
        self.output_Dir = dir_details.run_name
        # Set the label directory to "dir_details.label_dir"
        self.label_Dir = dir_details.label_dir
        # Get a list of all image IDs in the dataset, as well as dictionaries mapping IDs to file paths and labels
        self.all_ids, self.id_to_fp, self.id_to_lb = self.get_all_images_in_dataset()
        # Sort the list of IDs in ascending order
        self.all_ids.sort()
        # Define a list of all possible image numbers (i.e., the last two digits of the file names)
        all_images_no = ['2_10', '2_11', '2_12', '2_13', '2_14', 
                         '3_10', '3_11', '3_12', '3_13', '3_14', 
                         '4_10', '4_11', '4_12', '4_13', '4_14', '4_15',
                         '5_10', '5_11', '5_12', '5_13', '5_14', '5_15',
                         '6_7', '6_8', '6_9', '6_10', '6_11', '6_12', '6_13', '6_14', '6_15',
                         '7_7', '7_8', '7_9', '7_10', '7_11', '7_12', '7_13']
        # Set the random seed to 100
        random.seed(100)
        # Shuffle the list of image numbers randomly
        random.shuffle(all_images_no)
        # Select the first 14 image numbers as test images, the next 20 as training images, and the rest as validation images
        test_no = all_images_no[:14]
        train_no = all_images_no[14:14+20]
        val_no = all_images_no[14+20:]
        # Define empty lists for train, validation, and test image IDs
        self.train_ids = []
        self.val_ids = []
        self.test_ids = []
        # Loop over each image ID in "self.all_ids"
        for img_id in self.all_ids:
            # If the last two digits of the image ID are in the list of test numbers, add the ID to the test set
            if img_id[12:15] in test_no or img_id[12:16] in test_no:
                self.test_ids.append(img_id)
            elif img_id[12:15] in train_no or img_id[12:16] in train_no:
                self.train_ids.append(img_id)
            else:
                self.val_ids.append(img_id)

        self.save_graph = dir_details.save_graph
        data_stat = self.compute_dataset_stats() if STATS is None else STATS
        self.Dataset_mean = np.array(data_stat[0], dtype=np.float32)
        self.Dataset_std = np.array(data_stat[1], dtype=np.float32)

#This function returns the list of all image ids, and their corresponding file paths for both data and label folders

    def get_all_images_in_dataset(self):
        # Get all files in data and label directories
        data_folder = glob.glob(f"{self.data_Dir}/*", recursive=True)
        label_folder = glob.glob(f"{self.label_Dir}/*", recursive=True)

        # Create dictionaries of file paths for each image id
        scan_dic = {fp.split("/")[-1][:-4]: fp for fp in data_folder}
        label_dic = {fp.split("/")[-1][:-4]: fp for fp in label_folder}

        # Print number of images found
        print(f"Found {len(data_folder)} Images")

        # Return the list of all image ids, and their corresponding file paths for data and label folders
        return list(scan_dic.keys()), scan_dic, label_dic


    def compute_dataset_stats(self):
        # Prints a message indicating that dataset statistics are being computed
        print("Computing dataset mean and SD")

        # Initializes lists to store mean and standard deviation values for each image in the training set
        img_mean = []
        img_deviation = []

        # Loops over each image ID in the training set
        for img_id in self.train_ids:
            # Reads in the image using its file path
            image = io.imread(self.id_to_fp[img_id])
            
            # Computes the mean and standard deviation of the image pixels along the first and second axes (height and width)
            meanu = np.mean(image, axis=(0, 1))
            sigma = np.std(image, axis=(0, 1))
            
            # Appends the mean and standard deviation values for the current image to their respective lists
            img_mean.append(meanu)
            img_deviation.append(sigma)
        
        # Computes the mean and standard deviation of the mean and standard deviation values across all images in the training set
        Dataset_mean = np.mean(img_mean, axis=0)/255.0
        Dataset_deviation = np.mean(img_deviation, axis=0)/255.0
        
        # Prints the computed mean and standard deviation values
        print(f"Mean:{Dataset_mean}, SD: {Dataset_deviation} ")
        
        # Returns the dataset mean and standard deviation values
        return Dataset_mean, Dataset_deviation


    def get_standardized_image(self, img_id):
        # Load the image data from file
        img_data = io.imread(self.id_to_fp[img_id])
        # Apply data augmentation to the image data
        img_data = np.asarray(DataAug_Transformation(img_data))
        # Load the label data from file
        label_data = io.imread(self.id_to_lb[img_id.replace('RGB', 'label')])
        # Convert the label data into numeric encoding
        label_data = numeric_encoding(label_data)

        # Normalize the image data by dividing by 255.0
        normalized_dataset = (img_data) / 255.0
        # Standardize the image data using the dataset mean and standard deviation
        standardized_dataset = standardize_img(
            normalized_dataset, self.Dataset_mean, self.Dataset_std)
        
        # Return the original image data, the standardized image data, and the label data
        return img_data, standardized_dataset, label_data


    def process_next_sample(self, img_id, train_flag):
        # Check if training flag is True
        if train_flag:
            # Loop through the augmentation number
            for i in range(self.aug_no):
                # Obtain standardized image, image data, and label data
                img, image_data, label_data = self.get_standardized_image(img_id)
                
                # Convert image data and label data into a networkx graph, node features, region image, and slic partitioning
                nx_graph, node_feats, region_img, slic_partitioning = img2graph(image_data, label_data, self.num_node, self.boxiness_coef)
                
                # Check if saving graph is enabled
                if self.save_graph:
                    # Set save path for the processed image
                    save_path = f"{self.output_Dir}{os.sep}processed{os.sep}{img_id}_{i}"
                    
                    # Check if the save path exists, create it if it doesn't
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    
                    # Save the networkx graph to a json file
                    graph_io.save_networkx_graph(nx_graph, f"{save_path}{os.sep}{img_id}_nxgraph.json")
                    
                    # Save the region image as a numpy file
                    np.save(f"{save_path}{os.sep}{img_id}_region.npy", region_img)

        else:
            #It calls the get_standardized_image method and saves the output graph and images in the correct folder, if self.save_graph is True
            img, image_data, label_data = self.get_standardized_image(img_id)
            # calls img2graph method to get nx_graph, node_feats, region_img, and slic_partitioning
            nx_graph, node_feats, region_img, slic_partitioning = img2graph(
                image_data, label_data, self.num_node, self.boxiness_coef)

            # checks if self.save_graph is True
            if self.save_graph:
                # sets save_path to the processed folder path with the img_id
                save_path = f"{self.output_Dir}{os.sep}processed{os.sep}{img_id}"
                
                # checks if the path exists and creates one if it does not
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                # saves the networkx graph
                graph_io.save_networkx_graph(
                    nx_graph, f"{save_path}{os.sep}{img_id}_nxgraph.json")
                
                # saves the image data as a .npy file
                np.save(f"{save_path}{os.sep}{img_id}.npy", image_data)
                
                # saves the label data as a .tif file with the label keyword replaced by RGB
                io.imsave(f"{save_path}{os.sep}{img_id.replace('RGB', 'label')}.tif",
                        color_encoding(label_data), check_contrast=False)
                
                # saves the visualization image as a .jpg file
                io.imsave(f"{save_path}{os.sep}{img_id}_visulization.jpg", (mark_boundaries(
                    img, slic_partitioning)*255).astype(np.uint8), check_contrast=False)
                
                # saves the region image as a .npy file
                np.save(f"{save_path}{os.sep}{img_id}_region.npy", region_img)

            # returns img_id
            return img_id


    def run(self):
        # Define the number of threads to be used for concurrent execution
        N_THREADS = 8
        # Start measuring time
        t = time.time()
        
        # Start processing train samples using multithreading
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            # Create futures for each train image id
            futures = [executor.submit(self.process_next_sample, img_id, 1) for img_id in self.train_ids]
            # Print a message indicating that the threads are being set up and execution is starting
            print("Set up Threads, starting execution")
            # Wait for the futures to complete and handle exceptions, but ignore results
            for future in concurrent.futures.as_completed(futures):
                try:
                    img_id = future.result()
                except Exception as exc:
                    print(f"Thread generated exception {exc}")
                else:
                    continue

        # Start processing validation samples using multithreading
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            # Create futures for each validation image id
            futures = [executor.submit(self.process_next_sample, img_id, 0) for img_id in self.val_ids]
            # Print a message indicating that the threads are being set up and execution is starting
            print("Set up Threads, starting execution")
            # Wait for the futures to complete and handle exceptions, but ignore results
            for future in concurrent.futures.as_completed(futures):
                try:
                    img_id = future.result()
                except Exception as exc:
                    print(f"Thread generated exception {exc}")
                else:
                    continue

        # Start processing test samples using multithreading
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            # Create futures for each test image id
            futures = [executor.submit(self.process_next_sample, img_id, 0) for img_id in self.test_ids]
            # Print a message indicating that the threads are being set up and execution is starting
            print("Set up Threads, starting execution")
            # Wait for the futures to complete and handle exceptions, but ignore results
            for future in concurrent.futures.as_completed(futures):
                try:
                    img_id = future.result()
                except Exception as exc:
                    print(f"Thread generated exception {exc}")
                else:
                    continue

        # Print the average time taken for graph generation for each sample
        print("Time taken for avg graph generation is ", (time.time() - t)/len(self.all_ids), "seconds")



if __name__ == '__main__':

    #get details of the current run directory
    dir_details = get_run_details()
    print(dir_details)
    #populate hardcoded hyperparameters for data preprocessing
    hyper_params = populate_hardcoded_hyperparameters()
    #instantiate a DataPreprocessor object with the run directory details and hyperparameters
    gen = DataPreprocessor(dir_details, hyper_params)
    #run the data preprocessing pipeline
    gen.run()
    #create a run progress file in the run directory
    create_run_progress_file(
        f"{dir_details.run_name}{os.sep}{dir_details.run_name}.txt", hyper_params)

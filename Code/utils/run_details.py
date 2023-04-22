from collections import namedtuple

'''
Hyperparameters are set here. Can either hardcode or generate random ones. Hyperparameters are stored as named tuples.
Training the GNN has different hyperparameters than training the CNN.
'''

FullParamSet = namedtuple(
    "FullParamSet", 'run_name data_dir save_graph graph_dir out_dir label_dir')

#Starting parameter for storing the model with run_name
def get_run_details():

    run_name = "try7"
    data_dir = "./2_Ortho_RGB_patches_512"
    save_graph = True
    graph_dir = f"./{run_name}/processed"
    out_dir = ""
    label_dir = "./5_Labels_all_patches_512"

    dir_details = FullParamSet(
        run_name, data_dir, save_graph, graph_dir, out_dir, label_dir)

    return dir_details

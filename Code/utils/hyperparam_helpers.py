from collections import namedtuple

'''
Hyperparameters are set here. Can either hardcode or generate random ones. Hyperparameters are stored as named tuples.
Training the GNN has different hyperparameters than training the CNN.
'''

FullParamSet = namedtuple(
    "FullParamSet", 'n_nodes boxiness n_epochs in_feats out_classes lr lr_decay w_decay class_weights layer_sizes feature_dropout depth gat_heads gat_residuals')

DEFAULT_N_CLASSES = 6
DEFAULT_LR = 0.0001
DEFAULT_LR_DECAY = 0.98
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_FEATURE_DROPOUT = 0

DEFAULT_GNN_IN_FEATS = 30

#Hyperparameter used for the model training
def populate_hardcoded_hyperparameters():

    #Nodes define to form clusters in an input image
    n_nodes = 2000  # 2000
    #Defines the compactness of slic superpixels
    boxiness = 1
    #Initial epoch set 
    n_epochs = 200
    #input features 
    input_feats = DEFAULT_GNN_IN_FEATS
    #class weights
    class_weights = [0.05, 0.055, 0.06, 0.075, 0.55, 0.3]
    # layer_sizes = [512]*4
    depth = 4
    #Attention head specified
    attention_head = [6] * 10
    # layer_sizes = [32, 64, 128, 128, 128, 128, 128, 64, 32]

    #attention_head = [12, 12, 12, 12, 12, 12, 12, 12]
    #Layer size and neurons in each layer
    layer_sizes = [64, 128, 256, 512, 512, 256, 128, 64]

    # only relevant if model is GAT
    # att_heads = [4,4,3,3,4,4]
    # residuals = [False, False, False, True, True, False,
    #              False, False, True, False, False, True, False]

    #Residual boolean is set for internal connection
    residuals = [False, False, False, False, False,
                 False, False, False, False, False]

    hyperparams = FullParamSet(n_nodes, boxiness, n_epochs, input_feats,
                               DEFAULT_N_CLASSES, DEFAULT_LR, DEFAULT_LR_DECAY, DEFAULT_WEIGHT_DECAY,
                               class_weights, layer_sizes, DEFAULT_FEATURE_DROPOUT, depth, attention_head, residuals)

    return hyperparams

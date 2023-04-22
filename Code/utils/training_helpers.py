
'''
A collection of helper functions used while training a model.
'''

# save hyperparameters to a log file at a specified path
from pandas.core import accessor
from prettytable import PrettyTable
import matplotlib.pyplot as plt


def create_run_progress_file(fp, hp):

    with open(fp, "w") as f:
        table = PrettyTable(["Parameter", "Value"])
        table.add_row(["Num of Nodes", hp.n_nodes])
        table.add_row(["Compactness", hp.boxiness])
        table.add_row(["Epochs", hp.n_epochs])
        table.add_row(["Input Features", hp.in_feats])
        table.add_row(["Learning Rate", hp.lr])
        table.add_row(["Weight Decay", hp.w_decay])
        table.add_row(["LR Decay", hp.lr_decay])
        table.add_row(["Layer Sizes", hp.layer_sizes])
        table.add_row(["Att Heads", hp.gat_heads])
        f.write(table.get_string())


def chunk_dataset_into_folds(dataset, k):
    fold_size = len(dataset)//k
    folds = []
    for i in range(k):
        folds.append((i*fold_size, (i+1)*fold_size))
    return folds


def update_progress_file(fp, description, loss, dices):
    with open(fp, "a") as f:
        f.write(
            f"{description}\t{loss}\t{dices[0]}\t{dices[1]}\t{dices[2]}\t{dices[3]}\t{dices[4]}\t{dices[5]}\n")

# Pass in a model (which already contains the training data) and run it for a specified number of epochs.
# The model checkpoints its weights every couple of epochs.


def train(model, checkpoint_dir, n_epoch, run_name, val_dataset):
    train_loss = []
    val_loss = []
    train_f1_score = []
    val_f1_score = []
    lowest_loss = float("inf")
    patience = 0
    print("Training starts...")
    with open(checkpoint_dir + "loss.txt", "w") as f:
        # Loop through each epoch
        for i in range(1, n_epoch+1):
            # Train the model for one epoch and get the training loss and F1 score
            train_epoch_loss, train_f1 = model.run_epoch()
            train_loss.append(train_epoch_loss)
            train_f1_score.append(train_f1)

            # Evaluate the model on the validation dataset and get the loss and F1 score
            val_epoch_loss, val_f1 = model.evaluate(val_dataset)
            val_loss.append(val_epoch_loss)
            val_f1_score.append(val_f1)

            # Write the loss and F1 score to a file
            f.write("Epoch " + str(i) + " Training: loss= " + str(train_epoch_loss) +
                    " f1= " + str(train_f1) + " Validation: loss= " + str(val_epoch_loss) + " f1= " + str(val_f1) + "\n")
            f.flush()

            # Check if the validation loss has converged
            if((val_epoch_loss > lowest_loss) or (lowest_loss - val_epoch_loss < 0.005)):
                patience += 1
            else:
                patience = 0

            # If the validation loss has converged for 10 epochs, terminate training early
            if patience == 10:
                print("Terminated early due to converged validation loss")
                print(f"Ran for {i} epochs")
                break

            # If the validation loss is lower than the lowest validation loss seen so far,
            # save the model weights to a checkpoint file
            if val_epoch_loss < lowest_loss:
                lowest_loss = val_epoch_loss
                description = f"{run_name}"
                model.save_weights(checkpoint_dir, description)

#Plotting Graph
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    y1 = ax1.plot(range(1, len(train_loss)+1), train_loss,
                  color="green", label="train_loss")

    y2 = ax2.plot(range(1, len(train_f1_score)+1), train_f1_score,
                  color="red", label="train_F1-score")

    y3 = ax1.plot(range(1, len(val_loss)+1), val_loss,
                  color="blue", label="val_loss")

    y4 = ax2.plot(range(1, len(val_f1_score)+1),
                  val_f1_score, color="orange", label="val_F1-score")
    ax1.set_title(f"Plot for {run_name}")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("F1-score")
    leg = y1 + y2 + y3 + y4
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=0)
    plt.savefig(f"{checkpoint_dir}{run_name}.png", dpi=2000)

    print(f"Finished training...")

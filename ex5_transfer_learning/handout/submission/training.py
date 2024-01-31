from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy as loss_fun
from evaluation import evaluate_model


def train_model(
    model: nn.Module,
    train_loader,
    val_loader, 
    num_epochs,
    optimizer,
    device, 
    save_path=f"./ckpt/model.pt"
):
    """
    Feel free to change the arguments of this function - if necessary.
``
    Trains the model on the given dataset. Selects the best model based on the
    validation set and saves it to the given path. 
    Inputs: 
        model: The model to train [nn.Module]
        train_loader: The training data loader [DataLoader]
        val_loader: The validation data loader [DataLoader]
        num_epochs: The number of epochs to train for [int]
        optimizer: The optimizer [Any]
        best_of: The metric to use for validation [str: "loss" or "accuracy"]
        device: The device to train on [str: cpu, cuda, or mps]
        save_path: The path to save the model to [str]
    Output:
        Dictionary containing the training and validation losses and accuracies
        at each epoch. Also contains the epoch number of the best model.
    """

    #
    # You can put your training loop here
    #
    for i in tqdm(range(num_epochs)):
        loss = 0
        for image, label in train_loader:
            prediction = model(image)
            avg_loss = loss_fun(prediction, label)
            loss += avg_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model

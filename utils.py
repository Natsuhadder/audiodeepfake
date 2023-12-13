from bs4 import StopParsing
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from tqdm import trange



def is_early_stopping(scheduler,loss,optimizer,stopping_rate,elder_learning_rate,condition='lr'):
    scheduler.step(loss)
    learning_rate = optimizer.param_groups[0]['lr']
    if condition=='lr': 
        stop = (learning_rate < stopping_rate)
    if condition=='improvement':
        stop = (learning_rate != elder_learning_rate)
    else :
        return False
    return stop


def run_model(model, optimizer, loader, loss_function, device, timings=None, mode='train'):
    model.train() if mode == 'train' else model.eval()

    losses = []
    accuracies = []

    
    timer_starter = torch.cuda.Event(enable_timing=True)
    timer_ender = torch.cuda.Event(enable_timing=True)

    for data, target in loader:
        data_batch_shape = data.shape[0]
        data = torch.mean(data, dim=1).view(data_batch_shape, 1, 128)
        data, target = data.to(device), target.to(device)

        if timings is not None:
            timer_starter.record(torch.cuda.Stream())

        with torch.set_grad_enabled(mode == 'train'):
            output = model(data)
            loss = loss_function(output, target)

        if timings is not None:
            timer_ender.record(torch.cuda.Stream())
            torch.cuda.synchronize()
            curr_time = timer_starter.elapsed_time(timer_ender)
            timings.append(curr_time)

        preds = (torch.sigmoid(output) > 0.5) * 1
        accuracy = torch.mean((preds == target).type(torch.float))

        losses.append(loss.item())
        accuracies.append(accuracy.item())

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses, accuracies

from sklearn.metrics import accuracy_score, confusion_matrix

def run_model_eval(model, loader, device, timings=None):
    model.eval()
    model.to(device)

    all_preds = []
    all_true_labels = []

    timer_starter = torch.cuda.Event(enable_timing=True)
    timer_ender = torch.cuda.Event(enable_timing=True)
    
    for data, target in loader:
        data_batch_shape = data.shape[0]
        data = torch.mean(data, dim=1).view(data_batch_shape, 1, 128)
        data, target = data.to(device), target.to(device)
        
        with torch.no_grad():
            timer_starter.record(torch.cuda.Stream())
            output = model(data)
            timer_ender.record(torch.cuda.Stream())
            
            torch.cuda.synchronize(device)
            
            curr_time = timer_starter.elapsed_time(timer_ender)
            if timings is not None:
                timings.append(curr_time)
            
            preds = (torch.sigmoid(output) > 0.5)*1
            all_preds.extend(preds.cpu().numpy())
            all_true_labels.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true_labels = np.array(all_true_labels)

    if all_true_labels.ndim > 1 and all_true_labels.shape[1] > 1:
        all_true_labels = np.argmax(all_true_labels, axis=1)

    accuracy = accuracy_score(all_true_labels, all_preds)
    conf_matrix = confusion_matrix(all_true_labels, all_preds)

    return accuracy, conf_matrix





def training_model(model,optimizer,device,train_loader,test_loader,loss_function,scheduler,hparams,flops,timings=[],all_results_json=None,first_execution=True,early_stop=True):
    """Trains a deep learning model using specified hyperparameters, loaders, and device.

    Parameters:
    - model: The neural network model to be trained.
    - optimizer: The optimization algorithm used for training.
    - device: The device (CPU or GPU) on which the model will be trained.
    - train_loader: DataLoader for the training dataset.
    - test_loader: DataLoader for the test dataset.
    - loss_function: The loss function used for training.
    - scheduler: Learning rate scheduler.
    - hparams: An instance of HParams class containing hyperparameters.
    - flops: Floating Point Operations Per Second, an indication of model complexity.
    - timings (list, optional): A list to record timing information. Default is an empty list.
    - all_results_json (dict, optional): A dictionary to store all results. Default is None.
    - first_execution (bool, optional): Flag to indicate if this is the first execution for GPU warm-up. Default is True.
    - early_stop (bool, optional): Flag to enable early stopping. Default is True.

    The function performs training over a specified number of epochs, evaluating both training and validation loss and accuracy. It supports early stopping based 
    on validation accuracy improvement and records the performance metrics for each epoch.

    Returns:
    - train_loss_plot (list): List of training loss values for each epoch.
    - valid_loss_plot (list): List of validation loss values for each epoch.
    - train_acc_plot (list): List of training accuracy values for each epoch.
    - valid_acc_plot (list): List of validation accuracy values for each epoch.

    If `all_results_json` is provided, the function also appends the results of the training (model details, parameters, and training results) to this dictionary to get a summary
    of the model performance.
    """

    stopping_rate= hparams.stopping_rate
    learning_rate = hparams.learning_rate
    factor = hparams.factor
    patience = hparams.patience
    momentum = hparams.momentum 
    batch_size = hparams.batch_size
    

    train_loss_plot=[]
    valid_loss_plot=[]

    train_acc_plot=[]
    valid_acc_plot=[]

    epochs=[]
    def initialize_plots():
        """ Initialize plots for training and validation metrics. """
        return [], [], [], []

    def warm_up_gpu():
        """ Warm up GPU by running a few training iterations. """
        if first_execution:
            for _ in trange(3):
                print("WARMING UP FOR GPU")
                run_model(model=model, optimizer=optimizer, device=device, loader=train_loader, loss_function=loss_function, mode='train')

    def run_epoch(train_loader, test_loader):
        """ Run training and validation for one epoch. """
        train_losses, train_accuracy = run_model(model=model, optimizer=optimizer, device=device, loader=train_loader, loss_function=loss_function, timings=timings, mode='train')
        valid_losses, valid_accuracy = run_model(model=model, optimizer=optimizer, loader=test_loader, device=device, loss_function=loss_function, timings=timings, mode='eval')

        # Compute average losses and accuracies
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        avg_train_acc = sum(train_accuracy) / len(train_accuracy)
        avg_valid_acc = sum(valid_accuracy) / len(valid_accuracy)

        return avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc

    def check_early_stopping(valid_acc):
        """ Check and apply early stopping if required. """
        stop = False
        if early_stop:
            stop = is_early_stopping(scheduler=scheduler, loss=valid_acc, optimizer=optimizer, stopping_rate=stopping_rate, elder_learning_rate=learning_rate, condition='improvement')
            if stop:
                print(f"Early stopping for a learning rate = {optimizer.param_groups[0]['lr']}")
        return stop

    def append_results_json(stop):
        """ Append results to the JSON file if required. """
        if all_results_json is not None: 
            all_results_json.append({ 
                'model': {'architecture': str(model), 'FLOPS': flops},
                'parameters': {
                    'learning_rate': learning_rate,
                    'reduction factor': factor, 
                    'patience': patience, 
                    'momentum': momentum,
                    'batch_size': batch_size,
                    'device GPU' : torch.cuda.get_device_properties(device=device)
                },
                'results': {
                    'train_loss': train_loss_plot,
                    'valid_loss': valid_loss_plot,
                    'train_accuracy': train_acc_plot,
                    'valid_accuracy': valid_acc_plot,
                    'Number of epochs': epochs[-1],
                    'Early stop': stop,
                    'final learning_rate': optimizer.param_groups[0]['lr'],
                    'Time per epoch' :  np.sum(timings) / ((epochs[-1]+1)*1000)
                }
            })

    # Function starts here
    warm_up_gpu()

    print(f"Training of the model {str(model)} is starting ... :")
    train_loss_plot, valid_loss_plot, train_acc_plot, valid_acc_plot = initialize_plots()
    stop = False
    for epoch in trange(hparams.num_epochs):
        train_loss, valid_loss, train_acc, valid_acc = run_epoch(train_loader, test_loader)
        train_loss_plot.append(train_loss)
        valid_loss_plot.append(valid_loss)
        train_acc_plot.append(train_acc)
        valid_acc_plot.append(valid_acc)
        epochs.append(epoch)

        print(f"[Epoch {epoch + 1}/{hparams.num_epochs}] [Train Loss: {train_loss:.4f}] [Train Acc: {train_acc:.4f}] [Valid Loss: {valid_loss:.4f}] [Valid Acc: {valid_acc:.4f}]")

        stop = check_early_stopping(valid_acc)
        if stop:
            print(f"Early stopping for a learning rate = {optimizer.param_groups[0]['lr']}")
            break

    append_results_json(stop)
    print(f"Total time of training (only inference time) is {np.sum(timings)} and Time per epoch is : {np.sum(timings) / ((epochs[-1]+1)*1000)} s/epoch")

    return train_loss_plot, valid_loss_plot, train_acc_plot, valid_acc_plot,epochs
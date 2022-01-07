import torch
import time
import copy
import numpy as np
from sklearn import metrics as sk_metrics


def train_and_eval(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, num_epochs, early_stop, multiclass):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1score = 0.0
    best_prec = 0.0
    
    losses = []
    accs = []
    f1_scores, precision_scores = [], []
    stop = False

    val_losses, val_f1_scores = [], []
    current_epoch = num_epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{ num_epochs - 1}')
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            y_pred = []
            y_true = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                y_true.append(labels.cpu().numpy())
                y_pred.append(preds.cpu().numpy())
                running_loss += loss.item() * inputs.size(0)
                
            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = sk_metrics.accuracy_score(y_pred, y_true)
            if multiclass:
                f1_score = sk_metrics.f1_score(y_pred, y_true, average='weighted')
                precision = sk_metrics.precision_score(y_pred, y_true, average='weighted')
            else:
                precision = sk_metrics.precision_score(y_pred, y_true)
                f1_score = sk_metrics.f1_score(y_pred, y_true)

            print(f'{phase} Loss: {round(epoch_loss,4)} Acc: {round(epoch_acc,4)} F1Score: {round(f1_score, 4)}')

            if phase == 'train':
                losses.append(epoch_loss)
                scheduler.step()

            if phase == 'val':
                f1_scores.append(f1_score)
                accs.append(epoch_acc)
                val_f1_scores.append(f1_score)
                precision_scores.append(precision)

                if epoch >= 4 and early_stop:
                    if val_f1_scores[-1] <= np.mean(val_f1_scores[-4:-1]):
                        stop = True

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc and f1_score > best_f1score:
                best_acc = epoch_acc
                best_f1score = f1_score
                best_prec = precision
                best_model_wts = copy.deepcopy(model.state_dict())

        if stop:
            print(f"Stopped at epoch {epoch}.")
            current_epoch = epoch
            break

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {round(time_elapsed // 60,0)}m {round(time_elapsed % 60, 0)}s')
    print(f'Best val Acc: {round(best_acc,4)}')
    print(f'Best val f1score: {round(best_f1score, 4)}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = {
        "loss train": losses,
        "f1_score val": f1_scores,
        "acc val": accs,
        "prec val": precision_scores,
        "best val acc": round(best_acc, 4),
        "best val f1": round(best_f1score, 4),
        "best val precision": round(best_prec, 4)
    }
    return model, metrics, current_epoch


def train_and_eval_logo(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, num_epochs, early_stop, multiclass, logo, n_splits=0):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1score = 0.0
    best_prec = 0.0

    losses = []
    accs = []
    f1_scores, precision_scores = [], []
    stop = False

    val_losses, val_f1_scores = [], []
    current_epoch = num_epochs
    
    for fold, (train_index, val_index) in enumerate(logo):
        print(f'Patient {fold}/{ n_splits - 1}')
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{ num_epochs - 1}')
            print('-' * 10)
            # Each epoch has a training and validation phase
            for i, (inputs, labels) in enumerate(dataloaders['train']):
                if i in train_index and i not in val_index:
                    phase = 'train'
                    model.train()
                else:
                    phase = 'val'
                    model.eval()

                running_loss = 0.0
                y_pred = []
                y_true = []

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                y_true.append(labels.cpu().numpy())
                y_pred.append(preds.cpu().numpy())
                running_loss += loss.item() * inputs.size(0)

            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)

            epoch_loss = running_loss / dataset_sizes[phase]
            
            if epoch_loss == 0:
                break
            
            epoch_acc = sk_metrics.accuracy_score(y_pred, y_true)
            if multiclass:
                f1_score = sk_metrics.f1_score(y_pred, y_true, average='weighted')
                precision = sk_metrics.precision_score(y_pred, y_true, average='weighted')
            else:
                precision = sk_metrics.precision_score(y_pred, y_true)
                f1_score = sk_metrics.f1_score(y_pred, y_true)


            print(f'{phase} Loss: {round(epoch_loss,4)} Acc: {round(epoch_acc,4)} F1Score: {round(f1_score, 4)}')

            if phase == 'train':
                losses.append(epoch_loss)
                scheduler.step()

            if phase == 'val':
                f1_scores.append(f1_score)
                accs.append(epoch_acc)
                val_f1_scores.append(f1_score)
                precision_scores.append(precision)

                if epoch >= 4 and early_stop:
                    if val_f1_scores[-1] <= np.mean(val_f1_scores[-4:-1]):
                        stop = True

            # deep copy the model
            # CHECK BELLOW
            if phase == 'val' and epoch_acc > best_acc and f1_score > best_f1score:
                best_acc = epoch_acc
                best_f1score = f1_score
                best_prec = precision
                best_model_wts = copy.deepcopy(model.state_dict())

        if stop:
            print(f"Stopped at epoch {epoch}.")
            current_epoch = epoch
            break

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {round(time_elapsed // 60,0)}m {round(time_elapsed % 60, 0)}s')
    print(f'Best val Acc: {round(best_acc,4)}')
    print(f'Best val f1score: {round(best_f1score, 4)}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = {
        "loss train": losses,
        "f1_score val": f1_scores,
        "acc val": accs,
        "prec val": precision_scores,
        "best val acc": round(best_acc, 4),
        "best val f1": round(best_f1score, 4),
        "best val precision": round(best_prec, 4)
    }
    return model, metrics, current_epoch

import gc
import os

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

import wandb
from DataCleaning import process_data
from ModelClasses import NN, TestDataset, TrainDataset
from ModelFunctions import eval, test, train

if torch.cuda.is_available():
    DEVICE = 'cuda'
    DEVICE_N_WORKERS = 4
else:
    DEVICE = 'cpu'
    DEVICE_N_WORKERS = 0


def model_performace(model, X_train, X_test, y_train, y_test,
                     ljust_len=30):
    print('Training accuracy: {}'.format(
        "%.4f" % model.score(X_train, y_train)))
    print('Test accuracy: {}'.format(
        "%.4f" % model.score(X_test, y_test)))

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print('\n', report)


def checkpoint(file_path, model, optimizer, scaler, scheduler,
               epoch, train_acc, val_acc, precision, recall, f1):
    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scaler_state_dict': scaler.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
                  'train_accuracy': train_acc, 'val_accuracy': val_acc,
                  'precision': precision, 'recall': recall, 'f1': f1}
    torch.save(checkpoint, file_path)


def model_setting(input_size):
    # Create model
    model = NN(input_size=input_size,
               dropout_rate=wandb.config['dropout_rate']).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=wandb.config['init_lr'])
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=wandb.config['scheduler_factor'],
        patience=wandb.config['scheduler_patience'])

    # Save model
    model_arch = str(model)
    arch_file = open("model_arch.txt", "w")
    file_write = arch_file.write(model_arch)
    arch_file.close()
    wandb.save('model_arch.txt')

    return model, criterion, optimizer, scaler, scheduler


def model_training(model, criterion, optimizer, scaler, scheduler,
                   train_loader, val_loader, test_loader, y_test):
    torch.cuda.empty_cache()
    gc.collect()
    wandb.watch(model, log="all")

    best_score = 0

    for epoch in range(wandb.config['epochs']):
        print(f"\nEpoch {epoch+1}/{wandb.config['epochs']}")

        curr_lr = float(optimizer.param_groups[0]['lr'])
        train_loss, train_acc = train(model,
                                      train_loader,
                                      criterion,
                                      optimizer,
                                      scaler)
        val_loss, val_acc, precision, recall, f1 = eval(model,
                                                        val_loader,
                                                        criterion)
        scheduler.step(val_loss)

        if (np.allclose(precision, 0)):
            break

        print("\tTrain Acc {:.04f}%\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(
            train_acc*100, train_loss, curr_lr))
        print("\tVal Acc {:.04f}%\tVal Loss {:.04f}".format(
            val_acc*100, val_loss))
        print("\tVal Precision {:.04f}\tRecall {:.04f}\t\tF1 {:.04f}".format(
            precision, recall, f1))

        wandb.log({
            'lr': curr_lr,
            'train_acc': train_acc*100,
            'train_loss': train_loss,
            'val_acc': val_acc*100,
            'val_loss': val_loss,
            'val_precison': precision,
            'val_recall': recall,
            'val_f1': f1
        })

        epoch += 1

        if (precision > best_score):
            best_score = precision
            checkpoint(f'models/{wandb.run.id}_best_model.pt',
                       model, optimizer, scaler, scheduler,
                       epoch, train_acc, val_acc, precision, recall, best_score)
            print(f'Best model saved at epoch {epoch}')
            model_testing(model, test_loader, y_test)


def model_testing(model, test_loader, y_test, filename=None):
    y_pred = test(model, test_loader)
    report1 = f'''\
    accuracy: {accuracy_score(y_test, y_pred)}
    precision: {precision_score(y_test, y_pred)}
    recall: {recall_score(y_test, y_pred)}
    f1: {f1_score(y_test, y_pred)}
    '''
    report2 = classification_report(y_test, y_pred)
    if (filename is not None):
        with open(f'reports/{wandb.run.id}_{filename}.txt', 'w') as report:
            report.write(report1)
            report.write(report2)  # type: ignore
    else:
        print(report1)
        wandb.log({
            'test_acc': accuracy_score(y_test, y_pred)*100,
            'test_precison': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred)
        })


def downsample_data(X_train, y_train):
    X_downsampled, y_downsampled = sklearn.utils.resample(
        X_train[y_train == 0],
        y_train[y_train == 0],
        replace=False,
        n_samples=int(np.sum(y_train == 1) *
                      wandb.config['class_ratio']),
        random_state=42)  # type: ignore

    X_resampled = np.concatenate((X_downsampled,
                                  X_train[y_train == 1]))
    y_resampled = np.concatenate((y_downsampled,
                                  y_train[y_train == 1]))

    X_train, X_val, y_train, y_val = train_test_split(X_resampled,
                                                      y_resampled,
                                                      test_size=0.2,
                                                      random_state=42)

    return X_train, X_val, y_train, y_val


def sweep_train():
    with wandb.init():
        # Data processing
        X_train, X_test, y_train, y_test = process_data()
        X_train, X_val, y_train, y_val = downsample_data(X_train, y_train)

        train_data = TrainDataset(X=X_train, y=y_train)
        val_data = TrainDataset(X=X_val, y=y_val)
        test_data = TestDataset(X=X_test)

        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   num_workers=DEVICE_N_WORKERS,
                                                   batch_size=wandb.config['batch_size'],
                                                   pin_memory=True,
                                                   shuffle=True,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                                 num_workers=0,
                                                 batch_size=wandb.config['batch_size'],
                                                 pin_memory=True,
                                                 shuffle=False,
                                                 drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  num_workers=0,
                                                  batch_size=wandb.config['batch_size'],
                                                  pin_memory=True,
                                                  shuffle=False)

        # Training
        (model, criterion,
         optimizer, scaler, scheduler) = model_setting(train_data.n_feature)

        model_training(model, criterion, optimizer, scaler, scheduler,
                       train_loader, val_loader, test_loader, y_test)

        # Save results
        torch.save({'model_state_dict': model.state_dict(), },
                   f'models/{wandb.run.id}_final_model.pt')
        model_testing(model, test_loader, y_test, 'final')

        best_model = f'models/{wandb.run.id}_best_model.pt'
        if (os.path.exists(best_model)):
            checkpoint = torch.load(best_model, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_testing(model, test_loader, y_test, 'best')


if __name__ == '__main__':

    wandb.login(key="c3a06f318f071ae7444755a93fa8a5cbff1f6a86")

    sweep_config = {
        'method': 'random',
        'metric': {
            'goal': 'maximize',
            'name': 'val_precision'
        },
        'parameters': {
            'epochs': {'value': 50},
            'class_ratio': {
                'values': np.arange(1, 1.6, .1).tolist()
            },
            'batch_size': {
                'values': [64, 72, 84]
            },
            'init_lr': {
                'values': [1e-3, 75e-4, 5e-4]
            },
            'dropout_rate': {
                'values': [.1, .15, .2,]
            },
            'scheduler_factor': {
                'values': [.6, .7, .8]
            },
            'scheduler_patience': {
                'values': [2, 3]
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="map")
    wandb.agent(sweep_id, sweep_train, count=10)

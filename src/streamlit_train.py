import argparse
import os
import warnings
import numpy as np
import torch
import wandb
from augmentations import get_transform
from datasets.get_datasets import get_dataset_setting, get_datasets
from models import custom_resnet18 as resnet18
from models.custom_resnet18 import BasicBlock, ResNet
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm
import shutil
import itertools

warnings.filterwarnings("ignore", category=DeprecationWarning)

def train(model, train_loader, val_loader, current_epoch, metrics_storage, args):
    optimizer = SGD(
        model.parameters(),
        lr=args.lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    # to store metrics calculated for training as well as validation sets
    model.train()

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        images, class_labels, uq_idxs = batch
        images = images.to(args.device)
        class_labels = class_labels.to(args.device)

        # Extract features and output after linear classifier with model
        features, out = model(images)
        loss = torch.nn.CrossEntropyLoss()(out, class_labels)

        # Train acc
        _, preds = out.max(1)
        
        metrics_storage.update(mode='training', loss=loss.item(), size=class_labels.size(0), preds=preds.tolist(), labels=class_labels.tolist())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx == 2:
            break
    print(f'Train Epoch: {current_epoch} Avg Loss: {metrics_storage.metrics["train_loss"].avg} | Acc: {metrics_storage.metrics["train_acc"].avg}')
        
    # Evaluate on the validation set
    print('Evaluating on the disjoint validation set...')
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            images, class_labels, uq_idxs = batch
            images = images.to(args.device)
            class_labels = class_labels.to(args.device)

            # Extract features and output after linear classifier with model
            features, out = model(images)
            loss = torch.nn.CrossEntropyLoss()(out, class_labels)

            # Train acc
            _, preds = out.max(1)
            #acc = (pred == class_labels).float().mean().item()
            metrics_storage.update(mode='validation', loss=loss.item(), size=class_labels.size(0), preds=preds.tolist(), labels=class_labels.tolist())

            if batch_idx == 2:
                break
        print(f'Val Avg Loss: {metrics_storage.metrics["val_loss"].avg} | Acc: {metrics_storage.metrics["val_acc"].avg}')

def test(model, test_loader, args):
    model.eval()
    pred_probs = []
    labels = []
    preds = []

    print('Evaluating on the disjoint test set...')
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            images, class_labels, uq_idxs = batch
            images = images.to(args.device)
            class_labels = class_labels.to(args.device)

            # Extract features and output after linear classifier with model
            features, out = model(images)
            _, pred = out.max(1)
            preds += pred.tolist()
            pred_probs += out.tolist()
            labels += class_labels.tolist()
    return pred_probs, labels, preds
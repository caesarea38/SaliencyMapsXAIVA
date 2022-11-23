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
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import AverageMeter, get_mean_lr
import shutil

warnings.filterwarnings("ignore", category=DeprecationWarning)

def train(model, train_loader, val_loader, current_epoch, args):
    optimizer = SGD(
        model.parameters(),
        lr=args.lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_loss_record = AverageMeter()
    val_loss_record = AverageMeter()
    train_acc_record = AverageMeter()
    val_acc_record = AverageMeter()
    model.train()

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        images, class_labels, uq_idxs = batch
        images = images.to(args.device)
        class_labels = class_labels.to(args.device)

        # Extract features and output after linear classifier with model
        features, out = model(images)
        loss = torch.nn.CrossEntropyLoss()(out, class_labels)

        # Train acc
        _, pred = out.max(1)
        acc = (pred == class_labels).float().mean().item()
        train_acc_record.update(acc, pred.size(0))

        train_loss_record.update(loss.item(), class_labels.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx == 2:
            break
    print(f'Train Epoch: {current_epoch} Avg Loss: {train_loss_record.avg} | Acc: {train_acc_record.avg}')
        
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
            _, pred = out.max(1)
            acc = (pred == class_labels).float().mean().item()
            val_acc_record.update(acc, pred.size(0))

            val_loss_record.update(loss.item(), class_labels.size(0))
            if batch_idx == 2:
                break
        print(f'Val Avg Loss: {val_loss_record.avg} | Acc: {val_acc_record.avg}')
    
    metrics = {
        'train_loss': train_loss_record.avg,
        'train_acc': train_acc_record.avg,
        'val_loss': val_loss_record.avg,
        'val_acc': val_acc_record.avg 
    }
    return metrics
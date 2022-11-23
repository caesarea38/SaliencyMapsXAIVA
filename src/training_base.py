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

def train(model, train_loader, val_loader, args):
    optimizer = SGD(
        model.parameters(),
        lr=args.lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    for epoch in range(args.epochs):

        train_loss_record = AverageMeter()
        val_loss_record = AverageMeter()
        train_acc_record = AverageMeter()
        val_acc_record = AverageMeter()
        best_val_acc = 0
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

        print(f'Train Epoch: {epoch} Avg Loss: {train_loss_record.avg} | Acc: {train_acc_record.avg}')
        
        # Evaluate on the validation set
        print('Evaluating on the disjoint validation set...')
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
            print(f'Val Avg Loss: {val_loss_record.avg} | Acc: {val_acc_record.avg}')

        wandb.log({
            "epoch": epoch,
            "Train Loss":train_loss_record.avg,
            "Val Loss": val_loss_record.avg,
            "Train/LR":get_mean_lr(optimizer),
            "Train Accuracy":train_acc_record.avg,
            "Val Accuracy":val_acc_record.avg,
        })

        # Step schedule
        exp_lr_scheduler.step()

        # TODO: Development/Debugging
        #model_checkpoint_path = os.path.join('/xaiva_dev/saved_models/checkpoints', str(epoch))
        #model_best_path = os.path.join('/xaiva_dev/saved_models', 'best')

        model_checkpoint_path = os.path.join('/workspace/saved_models/checkpoints', str(epoch))
        model_best_path = os.path.join('/workspace/saved_models', 'best')

        torch.save(model.state_dict(), model_checkpoint_path + '/model.pt')
        print(f"model saved to {model_checkpoint_path}")

        if val_acc_record.avg > best_val_acc:
            best_val_acc = val_acc_record.avg
            print(f'Best Acc on validation set: {best_val_acc}...')
            torch.save(model.state_dict(), model_best_path + '/model_best.pt')
            print(f"Best model saved to {model_best_path}")

def test(model, test_loader, args):
    test_acc_record = AverageMeter()
    test_loss_record = AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            images, class_labels, uq_idxs = batch
            images = images.to(args.device)
            class_labels = class_labels.to(args.device)

            # Extract features and output after linear classifier with model
            features, out = model(images)
            loss = torch.nn.CrossEntropyLoss()(out, class_labels)

            # Test acc
            _, pred = out.max(1)
            acc = (pred == class_labels).float().mean().item()
            test_acc_record.update(acc, pred.size(0))

            test_loss_record.update(loss.item(), class_labels.size(0))
        print(f'Test Avg Loss: {test_loss_record.avg} | Acc: {test_acc_record.avg}')
        wandb.log({
            "Test Loss":test_loss_record.avg,
            "Test Accuracy":test_acc_record.avg,
        })


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sm_viz',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, cub200, inat21')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=0, type=int)
    parser.add_argument('--transform', type=str, default='pytorch-cifar', choices=['imagenet', 'pytorch-cifar'])
    parser.add_argument('--seed', default=1, type=int)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    args.device = torch.device('cuda:0')
    args = get_dataset_setting(args)
    print(args)
    wandb.login(key='3ad3d9b0536a3bb4aaf094b66988694b84787192')
    wandb.init(project="SMXAIVA", entity="mpss22")
    wandb.config = {
      "args": args,
    }

    # create folders for saving the model per epoch and the best model throughout the training based on some metric
    model_save_dir = os.path.join('/workspace', 'saved_models')
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
        os.mkdir(os.path.join(model_save_dir, 'checkpoints'))
        os.mkdir(os.path.join(model_save_dir, 'best'))
        for i in range(args.epochs):
            os.mkdir(os.path.join(model_save_dir, 'checkpoints', str(i)))
    
    # TODO: Development/Debugging
    #model_save_dir = os.path.join('/xaiva_dev', 'saved_models')
    #if not os.path.exists(model_save_dir):
    #shutil.rmtree(model_save_dir) 
    #os.mkdir(model_save_dir)
    #os.mkdir(os.path.join(model_save_dir, 'checkpoints'))
    #os.mkdir(os.path.join(model_save_dir, 'best'))
    #for i in range(args.epochs):
    #    print(i)
    #    os.mkdir(os.path.join(model_save_dir, 'checkpoints', str(i)))
    
    # ----------------------
    # BASE MODEL
    # ----------------------
    
    if args.model_name == 'resnet18':
        model = ResNet(BasicBlock, [2,2,2,2], args.num_classes).to(args.device)
    else:
        raise NotImplementedError

    # --------------------
    # TRANSFORMS
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, val_dataset, test_dataset = get_datasets(
        dataset_name=args.dataset_name,
        train_transform=train_transform,
        test_transform=test_transform,
        args=args
    )

    # --------------------
    # DATALOADERS
    # --------------------
    #train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, sampler=train_sampler, drop_last=True)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=True)
	
    # ----------------------
    # TRAIN
    # ----------------------
    train(
        model, 
        train_loader, 
        val_loader, 
        args
    )

    # ----------------------
    # TEST
    # ----------------------
    test(
        model,
        test_loader,
        args
    )

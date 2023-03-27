from project_config import cifar10_root, cifar100_root, cub_root
from datasets.cifar import CUSTOMCIFAR10, CUSTOMCIFAR100
from datasets.cub import CUSTOMCUB2011
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader

dataset_names = ['cub200', 'cifar10', 'cifar100', 'inat21']

def subsample_dataset(dataset, idxs):
        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]
        return dataset

def get_val_test_indices(dataset, val_split=0.3):
    classes = np.unique(dataset.targets)
    val_idxs = []
    test_idxs = []
    for cls in classes:
        cls_idxs = np.where(dataset.targets == cls)[0]
        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]
        val_idxs.extend(v_)
        test_idxs.extend(t_)
    return val_idxs, test_idxs

def get_dataset_setting(args):
    if args.dataset_name in ['cifar10', 'cifar100']:
        args.image_size = 32
        args.transform = 'pytorch-cifar'
        args.num_classes = 10 if args.dataset_name == 'cifar10' else 100
    elif args.dataset_name == 'cub200':
        args.image_size = 224
        args.num_classes = 200
        args.interpolation = 3
        args.crop_pct = 0.875
        args.transform = 'imagenet'
    return args

def get_dataloader(train_dataset, val_dataset, test_dataset, args):
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader
    
def get_datasets(dataset_name, train_transform, test_transform):
    """
    :return: train_dataset,
             val_dataset,
             test_dataset,
    """

    if dataset_name not in dataset_names:
        raise(ValueError)

    if dataset_name == 'cifar10':
        train_dataset = CUSTOMCIFAR10(root=cifar10_root, transform=train_transform, train=True, download=True)
        test_dataset = CUSTOMCIFAR10(root=cifar10_root, transform=test_transform, train=False, download=True)
    elif dataset_name == 'cifar100':
        train_dataset = CUSTOMCIFAR100(root=cifar100_root, transform=train_transform, train=True, download=True)
        test_dataset = CUSTOMCIFAR100(root=cifar100_root, transform=test_transform, train=False, download=True) 
    elif dataset_name == 'cub200':
        train_dataset = CUSTOMCUB2011(root=cub_root, transform=train_transform, train=True, download=True)        
        test_dataset = CUSTOMCUB2011(root=cub_root, transform=test_transform, train=False, download=True)        
    else:
        raise(NotImplementedError)

    val_idxs, test_idxs = get_val_test_indices(test_dataset)
    val_dataset = subsample_dataset(dataset=deepcopy(test_dataset), idxs=val_idxs)
    test_dataset = subsample_dataset(dataset=deepcopy(test_dataset), idxs=test_idxs)
    
    return train_dataset, val_dataset, test_dataset
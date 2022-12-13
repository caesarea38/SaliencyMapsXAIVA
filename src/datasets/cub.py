import os
import pandas as pd
import numpy as np
from copy import deepcopy
from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from project_config import cub_root

class CUSTOMCUB2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=True):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.pil_loader = pil_loader
        self.train = train

        if download:
            self.download()

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))
        self.targets = np.array([x for x in self.data['target']])
        self.data = np.array([x for x in self.data['filepath']])

    def load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def check_integrity(self):
        try:
            self.load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def download(self):
        import tarfile

        if self.check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        #sample = self.data.iloc[idx]
        #path = os.path.join(self.root, self.base_folder, sample.filepath)
        path = os.path.join(self.root, self.base_folder, sample)
        #target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        target = self.targets[idx] - 1
        img = self.loader(path)
        img_pil = pil_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, self.uq_idxs[idx], img_pil

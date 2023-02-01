from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np
import math
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor


class CUSTOMCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):
        super(CUSTOMCIFAR10, self).__init__(*args, **kwargs)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)

    def get_pil_image(self, item):
        img = self.data[item]
        img_pil = Image.fromarray(img)
        return img_pil, 0

class CUSTOMCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CUSTOMCIFAR100, self).__init__(*args, **kwargs)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)
    
    def get_pil_image(self, item):
        img = self.data[item]
        img_pil = Image.fromarray(img)
        return img_pil

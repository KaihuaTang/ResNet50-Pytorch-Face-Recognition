import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from PIL import Image
import os

class ImageData(Dataset):
    def __init__(self, root_path="CACD2000/", label_path="data/label.npy", name_path="data/name.npy", train_mode = "train"):
        """
        Initialize some variables
        Load labels & names
        define transform
        """
        self.root_path = root_path
        self.image_labels = np.load(label_path)
        self.image_names = np.load(name_path)
        self.train_mode = train_mode
        self.transform = {
            'train': transforms.Compose([                
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
 #               transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
 #               transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
            ]),
        }

    def __len__(self):
        """
        Get the length of the entire dataset
        """
        print("Length of dataset is ", self.image_labels.shape[0])
        return self.image_labels.shape[0]

    def __getitem__(self, idx):
        """
        Get the image item by index
        """
        image_name = os.path.join(self.root_path, self.image_names[idx])
        image = Image.open(image_name)
        image_label = self.image_labels[idx].astype(int) - 1
        transformed_img = self.transform[self.train_mode](image)
        sample = {'image':transformed_img, 'label':torch.from_numpy(image_label)}
        return sample
    
        

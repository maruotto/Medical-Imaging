# import the necessary packages
from torch.utils.data import Dataset
import cv2
import pandas as pd
import torch
import os
from skimage import io
import imutils
import numpy as np
from . import config
from .config import get_label_class, get_intensity_class

HEIGHT, WIDTH =  config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH
class MultiTaskDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        # store the image filepaths, the mask associated
        self.frame = pd.read_csv(csv_file, names=["Image", "Mask", "Label", "Intensity"])
        self.root_dir = root_dir

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get the image and the associated mask, and return them
        # grab the image path from the current index
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        image = io.imread(img_name)

        image = imutils.resize(image, width=WIDTH, height=HEIGHT)

        image2 = np.zeros((3, WIDTH, HEIGHT))
        image2[0, :, :] = image
        image2[1, :, :] = image
        image2[2, :, :] = image
        image = image2

        image = torch.as_tensor(image, dtype=torch.float32)

        label = self._get_label(idx)
        mask = self._get_mask(idx)
        intensity = self._get_intensity(idx)
        # return a tuple of the image and its label
        return (image, mask, label, intensity)

    def _get_label(self, idx):
        label = self.frame.iloc[idx, 2]

        label = torch.tensor(get_label_class(label), dtype=torch.long)

        return label

    def _get_intensity(self,idx):
        intensity = self.frame.iloc[idx, 3]
        #else not contemplated - it's a binary classification
        intensity = torch.tensor(get_intensity_class(intensity), dtype=torch.long)

        return intensity

    def _get_mask(self, idx):
        mask_name = os.path.join(self.root_dir, self.frame.iloc[idx, 1])
        mask = io.imread(mask_name)
        mask[mask == 255.0] = 1.0
        mask = imutils.resize(mask, width=WIDTH, height=HEIGHT)
        mask2 = np.zeros((1, WIDTH, HEIGHT))
        mask2[0, :, :] = mask
        mask = mask2

        mask = torch.as_tensor(mask, dtype=torch.float32)

        return mask
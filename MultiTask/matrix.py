# import the necessary packages
from training import config
import numpy as np
import torch
from training.dataset import MultiTaskDataset
import torchmetrics.functional as f
import pandas as pd
from torch.utils.data import DataLoader
import cv2
import argparse
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import seaborn as sns

color = {'BCE15-006': 'aqua',
         'BCE': 'forestgreen',
         'dice': 'darkslateblue',
         'BCE15-6': 'deepskyblue',
         'BCE15': 'lawngreen',
         'Dice15-6': 'purple',
         'Dice15-006': 'orchid',
         'dice15': 'crimson',
         }

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--letter", type=str, required=True, help="letter of the fold in capital case")
ap.add_argument("-p", "--path", type=str, required=False, help="path to output trained model", default="Confusions")
args = vars(ap.parse_args())

# load the image and mask filepaths in a sorted manner
LETTER = args["letter"]
OUTPUT = args["path"]

root_dir = config.IMAGE_DATASET_PATH
csv_file = config.test_dataset_path(LETTER)

intensity_labels = ['intermediate', 'positive']
label_labels = ['homogeneous', 'speckled', 'nucleolar', 'centromere', 'golgi', 'numem', 'mitsp']


def each_model(path, dir):
    print("[INFO] load up model " + dir + "...")
    unet = torch.load(MODEL_PATH).to(config.DEVICE)
    y_intensity = []
    preds_intensity = []
    y_label = []
    preds_label = []
    unet.eval()
    # turn off gradient tracking
    with torch.no_grad():
        for (x, y0, y1, y2) in testLoader:
            # send the input to the device
            x = x.to(config.DEVICE)
            preds = unet(x)

            y_label += [y1.cpu().item(), ]
            preds_label += [preds[1].argmax(1).to(config.DEVICE).cpu().item(), ]

            y_intensity += [y2.cpu().item(), ]
            preds_intensity += [
                torch.where(torch.sigmoid(preds[2].squeeze()) > torch.Tensor([config.THRESHOLD]).to(config.DEVICE), 1,
                            0).squeeze().cpu().item(), ]
    label_confusion = confusion_matrix(preds_label, y_label)
    intensity_confusion = confusion_matrix(preds_intensity, y_intensity)
    return label_confusion, intensity_confusion


def normalize_confusion(confusion_matrix):
    den = confusion_matrix.sum(axis=1)
    den = den.reshape(-1, 1)
    den = np.where(den==0, 1, den)
    return np.round((confusion_matrix / den)*100).astype(np.int8)


def create_pretty_confusion(confusion_matrix, intensity=True):
    plt.figure()
    sns.set()
    heatmap = sns.heatmap(confusion_matrix, annot=True, fmt="d",
                          xticklabels=intensity_labels if intensity else label_labels,
                          yticklabels=intensity_labels if intensity else label_labels, cmap="YlOrBr")
    # To save the heatmap as a file:
    return heatmap.get_figure()


if __name__ == '__main__':
    print("[INFO] loading up test image paths...")
    testData = MultiTaskDataset(csv_file=csv_file, root_dir=root_dir)
    testLoader = DataLoader(testData, shuffle=False, batch_size=1, pin_memory=config.PIN_MEMORY, num_workers=2)
    fig = plt.figure(1)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle=(0, (1, 10)))
    for dir in next(os.walk(config.BASE_OUTPUT))[1]:
        if not (dir == 'tmp' or dir == 'Confusions'or dir == 'Confusions Absolute' or dir == 'Rocs'):
            MODEL_PATH = os.path.join(config.BASE_OUTPUT, dir) + '/model' + LETTER
            label_confusion, intensity_confusion = each_model(MODEL_PATH, dir)
            fig = create_pretty_confusion(list(normalize_confusion(label_confusion)), False)
            plt.tight_layout()
            fig.savefig(os.path.join(config.BASE_OUTPUT, OUTPUT) + '/' + dir + 'label_heatmap' + LETTER + '.pdf')
            fig = None
            fig = create_pretty_confusion(list(normalize_confusion(intensity_confusion)), True)
            fig.savefig(os.path.join(config.BASE_OUTPUT, OUTPUT) + '/' + dir + 'intensity_heatmap' + LETTER + '.pdf')
            print('model ' + dir + ' success')



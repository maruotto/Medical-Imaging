# USAGE
# python3 predict.py

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

color = {'BCE15-006': 'aqua',
		'BCE': 'forestgreen',
		'dice': 'darkslateblue',
		'BCE15-6': 'deepskyblue',
		'BCE15': 'lawngreen',
		'Dice15-6': 'purple',
		'Dice15-006': 'orchid',
		'dice15': 'crimson',
		}

def plot_roc_curve(fpr, tpr, dir,auroc):
	plt.plot(fpr, tpr, color = color[dir], label=dir + ': ' + f'{auroc:.3f}')
    #plt.show()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--letter", type=str, required=True, help="letter of the fold in capital case")
ap.add_argument("-p", "--path", type=str, required=False, help="path to output trained model", default="output")
args = vars(ap.parse_args())

# load the image and mask filepaths in a sorted manner
LETTER = args["letter"]
OUTPUT = args["path"]


root_dir = config.IMAGE_DATASET_PATH
csv_file = config.test_dataset_path(LETTER)

def each_model(path, dir):
	print("[INFO] load up model " + dir +"...")
	unet = torch.load(MODEL_PATH).to(config.DEVICE)
	y_intensity_ = []
	preds_intensity_ = []
	unet.eval()
	# turn off gradient tracking
	with torch.no_grad():
		for (x, y0, y1, y2) in testLoader:
			# send the input to the device
			x = x.to(config.DEVICE)
			preds = unet(x)
			y_intensity_ += [y2.cpu().item(), ]
			preds_intensity_ += [torch.sigmoid(preds[2].squeeze()).cpu().item(), ]

	y_intensity = np.array(y_intensity_)
	preds_intensity = np.array(preds_intensity_)

	fpr, tpr, thresholds = roc_curve(y_intensity, preds_intensity)
	auroc = roc_auc_score(y_intensity, preds_intensity)
	print('AUROC: ', auroc)
	optimal_idx = np.argmax(tpr - fpr)
	optimal_threshold = thresholds[optimal_idx]
	print("Threshold optimal value is:", optimal_threshold)
	plot_roc_curve(fpr, tpr, dir, auroc=auroc)

	preds_intensity_old = torch.where(torch.Tensor(preds_intensity.squeeze()).to(config.DEVICE) > torch.Tensor([config.THRESHOLD]).to(config.DEVICE), 1, 0).squeeze().cpu()
	print('Intensity accuracy old',len(np.array(torch.where((torch.Tensor(preds_intensity_old) == torch.Tensor(y_intensity)))[0])) / len(testLoader))
	#print(confusion_matrix(preds_intensity_old, y_intensity))
	
	preds_intensity_new = torch.where(torch.Tensor(preds_intensity.squeeze()).to(config.DEVICE) > torch.Tensor([optimal_threshold]).to(config.DEVICE), 1, 0).squeeze().cpu()
	print('Intensity accuracy new', len(np.array(torch.where((torch.Tensor(preds_intensity_new) == torch.Tensor(y_intensity)))[0])) / len(testLoader))
	#print(confusion_matrix(preds_intensity_new, y_intensity))


if __name__ == '__main__':
	print("[INFO] loading up test image paths...")
	testData = MultiTaskDataset(csv_file=csv_file, root_dir=root_dir)
	testLoader = DataLoader(testData, shuffle=False, batch_size=1, pin_memory=config.PIN_MEMORY, num_workers=2)
	fig = plt.figure(1)
	plt.plot([0, 1], [0, 1], color='darkblue', linestyle=(0, (1, 10)))
	for dir in next(os.walk(config.BASE_OUTPUT))[1]:
		if not dir == 'tmp':
			MODEL_PATH = os.path.join(config.BASE_OUTPUT, dir) + '/model' + LETTER
			each_model(MODEL_PATH, dir)
			print('model ' + dir + ' success')

	plt.legend()
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')

	fig.savefig(os.path.join(config.BASE_OUTPUT, OUTPUT) + 'roc.pdf')



# python3 train.py --model output/model.pth --plot output/plot.png
# set the matplotlib backend so figures can be saved in the background
import matplotlib
import os
matplotlib.use("Agg")

# import the necessary packages
from training.dataset import MultiTaskDataset
from training.MultiTaskLoss import MultiTaskLoss
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import os
from training import config
from training.model import UNet
import torchmetrics.functional as f
from training.GradNorm import GradNorm

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--letter", type=str, required=True, help="letter of the fold in capital case")
ap.add_argument("-o", "--output", type=str, required=False, help="path to output trained model", default= "dice")
args = vars(ap.parse_args())

# load the image and mask filepaths in a sorted manner
LETTER = args["letter"]
OUTPUT = args["output"]
root_dir = config.IMAGE_DATASET_PATH
csv_file = 'dataset/crossValidationCSVs/fold' + LETTER + '_train.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the train and test datasets
trainData = MultiTaskDataset(csv_file=csv_file, root_dir=root_dir)
print(f"[INFO {time.asctime( time.localtime(time.time()) )}] found {len(trainData)} examples in the training set...")

torch.manual_seed(12345)
# create the training loader
trainLoader = DataLoader(trainData, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                         num_workers=2)

# calculate steps per epoch for training set
trainSteps = len(trainData) // config.BATCH_SIZE
    
# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=config.BATCH_SIZE)

# initialize the model
print(f"[INFO {time.asctime( time.localtime(time.time()) )}] initializing the model...")
model = UNet(config.NUM_CHANNELS,config.NUM_CLASSES).to(device)
    
# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=config.INIT_LR)
lossFn = MultiTaskLoss()

# initialize a dictionary to store training history
metrics = {"mask_loss": 0.0,
           "mask_acc": 0.0,
           "label_loss":0.0,
           "label_acc":0.0,
           "intensity_loss":0.0,
           "intensity_acc":0.0,
            "mask_train_loss": [],
           "label_train_loss": [],
           "intensity_train_loss": [],
           "mask_train_acc": [],
           "label_train_acc": [],
           "intensity_train_acc": []
}

# measure how long training is going to take
print(f"[INFO {time.asctime( time.localtime(time.time()) )}] training the network...")
startTime = time.time()

# set the model in training mode
model.train()
gradNorm = GradNorm(model, opt)

pred = None
# loop over our epochs
for e in range(0, config.NUM_EPOCHS):
#for e in range(0, 4):
    i=0
    # loop over the training set
    for x, y_mask, y_label, y_intensity in trainDataLoader:
        #print("iteration ", i, " of ", len(trainDataLoader))
        i +=1
        # send the input to the device
        x = x.to(device)
        y_mask = y_mask.to(device)
        y_label = [config.get_label_class(label) for label in y_label.tolist()]
        y_label = torch.Tensor(y_label).type(torch.uint8).to(device)
        y_intensity = torch.Tensor(y_intensity).to(device)
        # perform a forward pass and calculate the training loss
        pred = model(x) # the model returns a list of three elements - three predictions:  mask, label, intensity
        loss = lossFn(pred, y_mask, y_label, y_intensity)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        gradNorm.GradNorm_train(e, loss)
        normalize_coeff = 3 / torch.sum(model.weights.data, dim=0)
        model.weights.data = model.weights.data * normalize_coeff

        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        metrics["mask_loss"] = metrics["mask_loss"] + float(loss[0].cpu())
        metrics["label_loss"] = metrics["label_loss"] + float(loss[1].cpu())
        metrics["intensity_loss"] = metrics["intensity_loss"] + float(loss[2].cpu())

        where = torch.Tensor(torch.where(pred[0] > torch.Tensor([config.THRESHOLD]).to(device), 1, 0)).type(torch.uint8)
        metrics["mask_acc"] = metrics["mask_acc"] + float(f.dice(where.to(config.DEVICE), y_mask.type(torch.uint8)).cpu().item())
        metrics["label_acc"] = metrics["label_acc"] + float((pred[1].argmax(1) == y_label).type(torch.float).sum().cpu().item())
        metrics["intensity_acc"] = metrics["intensity_acc"] + float((pred[2].argmax(1) == y_intensity).type(torch.float).cpu().sum().item())

    # calculate the average training and validation loss

    metrics["mask_loss"] /= len(trainDataLoader.dataset)
    metrics["label_loss"] /= len(trainDataLoader.dataset)
    metrics["intensity_loss"] /= len(trainDataLoader.dataset)

    # calculate the training and validation accuracy
    metrics["mask_acc"] /= len(trainDataLoader.dataset)
    metrics["label_acc"] /= len(trainDataLoader.dataset)
    metrics["intensity_acc"] /= len(trainDataLoader.dataset)

  	# update our training history
    metrics["mask_train_acc"].append(metrics["mask_acc"])
    metrics["label_train_acc"].append(metrics["label_acc"])
    metrics["intensity_train_acc"].append(metrics["intensity_acc"])

  	# print the model training and validation information
    print(f"[INFO {time.asctime( time.localtime(time.time()) )}] ")
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
    print("Mask loss: {:.6f}, Mask accuracy: {:.4f}".format(metrics["mask_loss"], metrics["mask_acc"]))
    print("Label loss: {:.6f}, Label accuracy: {:.4f}".format(metrics["label_loss"], metrics["label_acc"]))
    print("Intensity loss: {:.6f}, Intensity accuracy: {:.4f}".format(metrics["intensity_loss"], metrics["intensity_acc"]))
        
# finish measuring how long training took
endTime = time.time()

plt.style.use("ggplot")
plt.figure(1)
plt.plot(metrics["mask_train_loss"], label="mask_train_loss")
plt.plot(metrics["label_train_loss"], label="label_train_loss")
plt.plot(metrics["intensity_train_loss"], label="intensity_train_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.BASE_OUTPUT+ OUTPUT+"/lossPlot"+LETTER)

# plot the training loss and accuracy

plt.figure(2)
plt.plot(metrics["mask_train_acc"], label="mask_train_acc")
plt.plot(metrics["label_train_acc"], label="label_train_acc")
plt.plot(metrics["intensity_train_acc"], label="intensity_train_acc")
plt.title("Training Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig(config.BASE_OUTPUT+ OUTPUT+"/accuracyPlot"+LETTER)




# serialize the model to disk
torch.save(model,config.BASE_OUTPUT + OUTPUT + "/model" +LETTER)
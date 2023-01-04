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
import sys
from training import config
from training.model import UNet
import torchmetrics.functional as f
from training.GradNorm import GradNorm
import math

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--letter", type=str, required=True, help="letter of the fold in capital case")
ap.add_argument("-o", "--output", type=str, required=False, help="path to output trained model", default="BCE")
ap.add_argument("-e", "--epochs", type=str, required=False, help="number of epochs to add", default=5)
ap.add_argument("-m", "--model", type=str, required=False, help="path to input pretrained model", default="BCE15")

args = vars(ap.parse_args())

# load the image and mask filepaths in a sorted manner
LETTER = args["letter"]
OUTPUT = args["output"]
NUM_EPOCHS = int(args["epochs"])
root_dir = config.IMAGE_DATASET_PATH
MODEL_PATH = os.path.join(config.BASE_OUTPUT, args["model"]) +  "/model" + LETTER
csv_file = 'dataset/crossValidationCSVs/fold' + LETTER + '_train.csv'

device = config.DEVICE

# create the train and test datasets
trainData = MultiTaskDataset(csv_file=csv_file, root_dir=root_dir)
print(f"[INFO {time.asctime(time.localtime(time.time()))}] found {len(trainData)} examples in the training set...")

torch.manual_seed(12345)
# create the training loader
trainLoader = DataLoader(trainData, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                         num_workers=2)

# calculate steps per epoch for training set
trainSteps = len(trainData) // config.BATCH_SIZE

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=config.BATCH_SIZE)

# initialize the model
print(f"[INFO {time.asctime(time.localtime(time.time()))}] initializing the model...")
model = unet = torch.load(MODEL_PATH).to(config.DEVICE) #UNet(config.NUM_CHANNELS, config.NUM_CLASSES).to(device)

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=config.INIT_LR)
lossFn = MultiTaskLoss()

# initialize a dictionary to store training history
metrics = {"mask_loss": np.array([]),
           "mask_acc": np.array([]),
           "label_loss": np.array([]),
           "label_acc": np.array([]),
           "intensity_loss": np.array([]),
           "intensity_acc": np.array([]),
           "mask_train_loss": np.array([]),
           "label_train_loss": np.array([]),
           "intensity_train_loss": np.array([]),
           "mask_train_acc": np.array([]),
           "label_train_acc": np.array([]),
           "intensity_train_acc": np.array([]),
           }

# measure how long training is going to take
print(f"[INFO {time.asctime(time.localtime(time.time()))}] training the network...")
startTime = time.time()

# set the model in training mode
model.train()
gradNorm = GradNorm(model, opt)

pred = None
# loop over our epochs
for e in range(0, NUM_EPOCHS):
#for e in range(0, 1):
    print('0.06 as alpha\t0.001 learning rate\t8 batch size\nSTART NEW EPOCH')
    i = 0
    metrics ["mask_loss"] =  np.array([]).astype(np.double)
    metrics ["mask_acc"] =  np.array([])
    metrics ["label_loss"] =  np.array([])
    metrics["label_acc"] = np.array([])
    metrics["intensity_loss"] = np.array([])
    metrics["intensity_acc"] =  np.array([]).astype(np.double)

    # loop over the training set
    for x, y_mask, y_label, y_intensity in trainDataLoader:
        print("iteration ", i, " of ", len(trainDataLoader))
        i += 1
        sys.stdout.flush()
        # send the input to the device
        x = x.to(device)
        y_mask = y_mask.to(device)
        y_label = torch.Tensor(y_label).type(torch.uint8).to(device)
        y_intensity = torch.Tensor(y_intensity).to(device)
        # perform a forward pass and calculate the training loss
        pred = model(x)  # the model returns a list of three elements - three predictions:  mask, label, intensity
        loss = lossFn(pred, y_mask, y_label, y_intensity)   
        #loss is a tensor of three elements where each element is the task loss associated to one batch
        
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        gradNorm.GradNorm_train(e, loss)
        normalize_coeff = 3 / torch.sum(model.weights.data, dim=0)
        model.weights.data = model.weights.data * normalize_coeff

        # update the loss metrics for mask, label and intensity
        metrics["mask_loss"] = np.append(metrics["mask_loss"], float(loss[0].cpu()))
        metrics["label_loss"] = np.append(metrics["label_loss"], float(loss[1].cpu()))
        metrics["intensity_loss"] = np.append(metrics["intensity_loss"], float(loss[2].cpu()))
        
        

        # calculate the number of correct predictions
        '''
        l = 0
        dice_score = []
        for elem in pred[0]:
            pos_prob = torch.sigmoid(elem)
            pos_prob = torch.Tensor(torch.where(pos_prob > torch.Tensor([config.THRESHOLD]).to(device), 1, 0)).type(
                torch.uint8)
            dice = f.dice(pos_prob.to(config.DEVICE), y_mask[l].type(torch.uint8)).item()
            dice_score.append(dice)
            l += 1
            dice_score_batch = torch.Tensor(dice_score).mean().to(config.DEVICE)
        '''

        dice_score_batch = torch.sigmoid(pred[0])
        where = torch.Tensor(torch.where(dice_score_batch> torch.Tensor([config.THRESHOLD]).to(device), 1, 0)).type(torch.uint8)
        dice_score_batch = f.dice(where.to(config.DEVICE), y_mask.type(torch.uint8).to(config.DEVICE)).cpu().item()
        metrics["mask_acc"] = np.append(metrics["mask_acc"], dice_score_batch)
        metrics["label_acc"] = np.append(metrics["label_acc"], len(torch.where((pred[1].argmax(1) == y_label).cpu())[0]))
        intensity_pred = torch.sigmoid(pred[2].squeeze())
        where = torch.Tensor(torch.where(intensity_pred > torch.Tensor([config.THRESHOLD]).to(device), 1, 0)).type(torch.uint8)
        metrics["intensity_acc"] = np.append(metrics["intensity_acc"], len(torch.where((where.squeeze() == y_intensity).cpu())[0]))
    # calculate the average training and validation loss

    # update our training history
    metrics["mask_train_acc"] = np.append(metrics["mask_train_acc"], metrics["mask_acc"].mean().item())
    metrics["label_train_acc"] = np.append(metrics["label_train_acc"], metrics["label_acc"].sum().item() / (config.BATCH_SIZE * len(trainDataLoader)))
    metrics["intensity_train_acc"] = np.append(metrics["intensity_train_acc"], metrics["intensity_acc"].sum().item() / (config.BATCH_SIZE * len(trainDataLoader)))

    '''
    print('old mask loss: ', metrics["mask_loss"].sum().item() / (config.BATCH_SIZE * len(trainDataLoader)))
    print('new loss: ', metrics["mask_loss"].mean().item())
    print('mul: ', metrics["mask_loss"].sum().item() * 8 / (config.BATCH_SIZE * len(trainDataLoader)))
    print()

    print('old label loss: ', metrics["label_loss"].sum().item() /  (config.BATCH_SIZE * len(trainDataLoader)))
    print('new loss: ', metrics["label_loss"].sum().item() /  len(trainDataLoader))
    print('mul: ', metrics["label_loss"].sum().item() * 8/  (config.BATCH_SIZE * len(trainDataLoader)))
    print()

    print('old intensity loss: ',  metrics["intensity_loss"].sum().item() /  (config.BATCH_SIZE * len(trainDataLoader)))
    print('new loss: ', metrics["intensity_loss"].sum().item() /  len(trainDataLoader))
    print('mul: ', metrics["intensity_loss"].sum().item() * 8/  (config.BATCH_SIZE * len(trainDataLoader)))
    print()
    '''
    metrics["mask_train_loss"] = np.append(metrics["mask_train_loss"], metrics["mask_loss"].mean().item())
    metrics["label_train_loss"] = np.append(metrics["label_train_loss"], metrics["label_loss"].sum().item() /  len(trainDataLoader))
    metrics["intensity_train_loss"] = np.append(metrics["intensity_train_loss"],metrics["intensity_loss"].sum().item() / len(trainDataLoader))

    # print the model training and validation information
    print(f"[INFO {time.asctime(time.localtime(time.time()))}] ")
    print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
    print("Mask loss: {:.6f}, Mask accuracy: {:.4f}".format(metrics["mask_train_loss"][-1],
                                                            metrics["mask_train_acc"][-1]))
    print("Label loss: {:.6f}, Label accuracy: {:.4f}".format(metrics["label_train_loss"][-1],
                                                              metrics["label_train_acc"][-1]))
    print("Intensity loss: {:.6f}, Intensity accuracy: {:.4f}".format(metrics["intensity_train_loss"][-1],
                                                                      metrics["intensity_train_acc"][-1]))
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
plt.savefig(config.BASE_OUTPUT + OUTPUT + "/lossPlot" + LETTER)

# plot the training loss and accuracy

plt.figure(2)
plt.plot(metrics["mask_train_acc"], label="mask_train_acc")
plt.plot(metrics["label_train_acc"], label="label_train_acc")
plt.plot(metrics["intensity_train_acc"], label="intensity_train_acc")
plt.title("Training Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig(config.BASE_OUTPUT + OUTPUT + "/accuracyPlot" + LETTER)

# serialize the model to disk
torch.save(model, config.BASE_OUTPUT + OUTPUT + "/model" + LETTER)
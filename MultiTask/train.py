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
from training import config
from training.model import UNet
import torchmetrics.functional as f

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--letter", type=str, required=True, help="path to output trained model")
args = vars(ap.parse_args())

# load the image and mask filepaths in a sorted manner
LETTER = args["letter"]
root_dir = config.IMAGE_DATASET_PATH
csv_file = 'dataset/crossValidationCSVs/fold' + LETTER + '_train.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the train and test datasets
trainData = MultiTaskDataset(csv_file=csv_file, root_dir=root_dir)
print(f"[INFO] found {len(trainData)} examples in the training set...")

torch.manual_seed(12345)
# create the training loader
trainLoader = DataLoader(trainData, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                         num_workers=2)

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

# calculate steps per epoch for training set
trainSteps = len(trainData) // config.BATCH_SIZE
    
# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)

# initialize the model
print("[INFO] initializing the model...")
model = UNet(config.NUM_CHANNELS,config.NUM_CLASSES).to(device)
    
# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = MultiTaskloss()

# initialize a dictionary to store training history
H = {"train_loss": [], "mask_acc": [], "label_train_acc": [], "intensity_train_acc": []}

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

# set the model in training mode
model.train()
gradNorm = GradNorm(model, opt)
maskTrainCorrect = 0
labelTrainCorrect = 0
intensityTrainCorrect = 0
# loop over our epochs
for e in range(0, EPOCHS):
#for e in range(0, 1):

    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    valCorrect = 0
    i=0
    # loop over the training set
    for x, y_mask, y_label, y_intensity in trainDataLoader:
       # if i%5 == 0:
        print("iteration ", i, " of ", len(trainDataLoader))
        i +=1
        # send the input to the device
        x = x.to(device)
        y_mask = y_mask.to(device)
        y_label = [config.get_label_class(label) for label in y_label.tolist()]
        y_label = torch.Tensor(y_label).type(torch.uint8).to(device)
        y_label = y_label.to(device)
        y_label = y_label.to(device)
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
        #totalTrainLoss += loss

        where = torch.Tensor(torch.where(pred[0] > torch.Tensor([config.THRESHOLD]).to(device), 1, 0)).type(torch.uint8)

        maskTrainCorrect += f.dice(where.to(config.DEVICE), y_mask.type(torch.uint8)).item()
        labelTrainCorrect += (pred[1].argmax(1) == y_label).type(torch.float).sum().item()
        intensityTrainCorrect += (pred[2].argmax(1) == y_intensity).type(torch.float).sum().item()

    # calculate the average training and validation loss
    #avgTrainLoss = totalTrainLoss / len(trainDataLoader.dataset)
    #avgValLoss = totalValLoss / len(trainDataLoader.dataset)

    # calculate the training and validation accuracy
    maskTrainCorrect = maskTrainCorrect / len(trainDataLoader.dataset)
    labelTrainCorrect = labelTrainCorrect / len(trainDataLoader.dataset)
    intensityTrainCorrect = intensityTrainCorrect / len(trainDataLoader.dataset)

  	# update our training history
    #H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["mask_train_acc"].append(maskTrainCorrect)
    H["label_train_acc"].append(labelTrainCorrect)
    H["intensity_train_acc"].append(intensityTrainCorrect)
    
  	# print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    #print("Train loss: {:.6f}, Mask accuracy: {:.4f}".format(avgTrainLoss, maskTrainCorrect))
    print("Label loss: {:.6f}, Intensity accuracy: {:.4f}".format(labelTrainCorrect, intensityTrainCorrect))
        
# finish measuring how long training took
endTime = time.time()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
#plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["mask_train_acc"], label="mask_acc")
plt.plot(H["label_train_acc"], label="label_train_acc")
plt.plot(H["intensity_train_acc"], label="intensity_train_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.BASE_OUTPUT+"dice/plot"+LETTER)

# serialize the model to disk
torch.save(model,config.BASE_OUTPUT+ "dice/model"+LETTER)

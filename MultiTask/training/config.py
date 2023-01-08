# import the necessary packages
import torch
import os

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join("dataset", "train")

def test_dataset_path(letter):
	fold_name = "fold" + letter + "_test.csv"
	return os.path.join("dataset/crossValidationCSVs", fold_name)


def train_dataset_path(letter):
	fold_name = "fold" + letter + "_train.csv"
	return os.path.join("dataset/crossValidationCSVs", fold_name)

# define the train - test split
TRAIN_SPLIT = 0.8

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 3
NUM_CLASSES = 1

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.0001
NUM_EPOCHS = 15#30
BATCH_SIZE = 8

# define the input image dimensions
INPUT_IMAGE_WIDTH = 384
INPUT_IMAGE_HEIGHT = 384

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output/"

#define alpha parameter of gradnorm
ALPHA = 6

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "model")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

def model_path(folder):
	return os.path.join(BASE_OUTPUT, folder,"unet_tgs_salt.pth")

def plot_path(folder):
	return os.path.join(BASE_OUTPUT, folder,"plot.png")

def get_label_class(label):
	if label == 'homogeneous ':
		return 0
	elif label == 'speckled ':
		return 1
	elif label == 'nucleolar ':
		return 2
	elif label == 'centromere ':
		return 3
	elif label == 'golgi ':
		return 4
	elif label == 'numem ':
		return 5
	elif label ==  'mitsp ':
		return 6

def get_intensity_class(intensity):
	if intensity == 'intermediate ' or intensity == 'intermediate':
		return 0
	elif intensity == 'positive ' or intensity == 'positive':
		return 1

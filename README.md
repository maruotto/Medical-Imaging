# Medical-Imaging
Medical Imaging Project, a.a. 2022-2023, group 1

Upon adapting the U-Net network to perform joint intensity classification and specimen segmentation on HEp-2 cells, the network has been further modified by changing its backbone with the ResNet34 network, and adopting the Gradient Normalization algorithm for training. This project evaluates the performances obtained on the HEP2 dataset using a combined loss - one for each task.

For the first approach, the complete loss is defined as so:
  - binary cross entropy, for the intensity classification of the images;
  - cross entropy, for the label classification of the images;
  - dice loss, for the segmentation of the images.

For the second approach, the loss is, instead, defined as:
  - binary cross entropy, for the intensity classification of the images;
  - cross entropy, for the label classification of the images;
  - binary cross entropy, for the segmentation of the images.
________________________________________________________________________________________________________________________________________________________________________
Networks are too big to be uploaded on GitHub, therefore there is a [link](https://drive.google.com/drive/folders/1ucfmfyq5BV9XNvvPFlwA5JQjxJ__9RKQ?usp=share_link) to a Google Drive Folder accessible to members of Università degli Studi di Salerno

Scripts are organized in the following way:
  - train.py, predict.py, roc.py and matrix.py work on a single fold.
  - train.sh, predicts.sh, rocs.sh and matrixes.sh work on all the five folds.

The training folder contains utility files for the training phase.

To train execute ```nohup ./train.sh OUPUT_FOLDER >& output/trainLog.log &```.

To predict results execute ```nohup ./predicts.sh INPUT_FOLDER >& output/predict.log &```.

OUTPUT_FOLDER and INPUT_FOLDER must be inside 'output' folder, defined in config.py.

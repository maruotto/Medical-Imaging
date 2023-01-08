# Medical-Imaging
Medical Imaging Project, a.a. 2022-2023, group 1

Networks are too big to be uploaded on GitHub, therefore there is a [link](https://drive.google.com/drive/folders/1ucfmfyq5BV9XNvvPFlwA5JQjxJ__9RKQ?usp=share_link) to a Google Drive Folder accessible to members of Università degli Studi di Salerno

Scripts are organized in the following way:
  - train.py, predict.py, roc.py and matrix.py work on a single fold.
  - train.sh, predicts.sh, rocs.sh work on all the five folds.

The training folder contains utility files for the training phase.

To train execute ```nohup ./train.sh OUPUT_FOLDER >& output/trainLog.log &```
To predict results execute ```nohup ./predict.sh INPUT_FOLDER >& output/predict.log &```
OUTPUT_FOLDER and INPUT_FOLDER must be inside 'output' folder, defined in config.py

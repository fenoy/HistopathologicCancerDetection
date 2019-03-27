# Hitopathologic Cancer Detection

This is the code for a Kaggle competition which you can find all the information [here](https://www.kaggle.com/c/histopathologic-cancer-detection/). It consists in a binary classification problem which objective is to classify small patch images according to if there is a metastasic cancer in it.

## Model

I have implemented two different models to make the predictions (they can be found in /src/model.py). First one is a simple Convolutional Neural Network. The second one is a SE-ResNet which you can read about [here](https://arxiv.org/abs/1512.03385) and [here](https://arxiv.org/pdf/1709.01507.pdf).

I ended up using a pretrained version of the SE-ResNet which can be found [here](https://pypi.org/project/pretrainedmodels/). I changed the last linear layer to be a sequential fully-connected in order to fit the pretrained model to my dataset.

## Training

- Epochs: 20
- Batch Size: 128
- Loss: Binary Cross Entropy
- Optimizer: Adam
- Learning Rate: 0.007 reduced by a factor 0.5 each 2 epochs

A validation set of size was created from the original training set, spliting it in sizes (0.2 val/ 0.8 test). Validation was done each epoch to track overfitting. At the end of each epoch, if the accuracy of the validation set was better than in the previous epoch, the model was saved.

## Test

The test was done using the best model (chosen according to AUC-ROC metrics). It has also been used the Test Time Augmentation tecnique to imporove the accuracy of the results.

## To run the code

- Create folder for checkpoint and data.

~~~~
mkdir checkpoint
mkdir data
~~~~

- Go to data folder and download the data from Kaggle (you need the kaggle api).

~~~~
cd data
kaggle competitions download -c histopathologic-cancer-detection
~~~~

- Go to src folder and run data.py to extract the data.

~~~~
cd data
python3 data.py
~~~~

- Train the model.

~~~~
cd data
python3 train.py
~~~~

- Test the model saved in checkpoint folder.

~~~~
python3 test.py ../checkpoint/epp18AUC0.997.pt
~~~~

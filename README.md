# iDPP2022best
Code for the IDPP@CLEF 2022 best of follow-up paper

## How To Use
Create a venv and install all the requirements:

    pip install -r requirements.txt

## Description
The notebooks called: 'lisbona_preprocessing.ipynb' and 'torino_preprocessing.ipynb' contain the preprocessing of each dataset from each center.

The notebooks called: 'static.ipynb' and 'survey.ipynb' contain the processing to obtain the two datasets with only static features and the one with ALSFRS. However, these two notebooks are here only to show the process because the same processing is done inside the 'tran_test.py' script.

If you like to replicate the research, you have to go inside the 'main.py' and select which dataset create and then launch:

    python main.py

It will retrain the models and test the best models from the GridSearch.

## Code Features
Inside the folder utils there are two files in order to run all the code. Inside 'utils_prediction.py' there are some commented lines. You can save the models or the results from the Cross Validation decommenting that lines.
import numpy as np
import pandas as pd
from utils.utils import convert_ref, static_preprocessing, one_alsfrs
from utils.utils_prediction import task_death, task_niv, task_peg, task_niv_death, task_peg_death
pd.options.mode.chained_assignment = None

data_torino = pd.read_excel('preprocessed_data/torino.xlsx')
data_torino_survey = pd.read_excel('preprocessed_data/torino.xlsx', sheet_name=1)
data_lisbona = pd.read_excel('preprocessed_data/lisbona.xlsx')
data_lisbona_survey = pd.read_excel('preprocessed_data/lisbona.xlsx', sheet_name=1)

data_lisbona_survey['REF'] = convert_ref(data_lisbona_survey)
data_lisbona['REF'] = convert_ref(data_lisbona)

def main_survey():
    data_merged = static_preprocessing(data_torino, data_lisbona)
    data_merged = one_alsfrs(data_merged, data_torino_survey, data_lisbona_survey, data_torino, data_lisbona)
    data_merged['Gender']=data_merged['Gender'].apply(lambda x: 0 if x == 1.0 else 1)

    print('FIRST TASK: DEATH PREDICTION\n')
    task_death(data_merged)
    print('\n')

    print('SECOND TASK: NIV PREDICTION\n')
    task_niv(data_merged)
    print('\n')

    print('THIRD TASK: PEG PREDICTION\n')
    task_peg(data_merged)
    print('\n')

    print('FOURTH TASK: NIV or DEATH PREDICTION\n')
    task_niv_death(data_merged)
    print('\n')

    print('FIFTH TASK: PEG or DEATH PREDICTION\n')
    task_peg_death(data_merged)
    print('\n')

def main_static():
    data_merged = static_preprocessing(data_torino, data_lisbona)
    data_merged['Status']=data_merged['Status'].apply(lambda x: False if x == 1.0 else True)
    data_merged['Gender']=data_merged['Gender'].apply(lambda x: 0 if x == 1.0 else 1)

    print('FIRST TASK: DEATH PREDICTION\n')
    task_death(data_merged)
    print('\n')

    print('SECOND TASK: NIV PREDICTION\n')
    task_niv(data_merged)
    print('\n')

    print('THIRD TASK: PEG PREDICTION\n')
    task_peg(data_merged)
    print('\n')

    print('FOURTH TASK: NIV or DEATH PREDICTION\n')
    task_niv_death(data_merged)
    print('\n')

    print('FIFTH TASK: PEG or DEATH PREDICTION\n')
    task_peg_death(data_merged)
    print('\n')

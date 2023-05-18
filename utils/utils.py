import logging
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_censored


logger = logging.getLogger()


def dataframe_creation(dataframe): 

    #dictionary_results = {}
    
    #for column in dataframe.columns[1:-2]:

    results = pd.DataFrame(columns=['REF'])
    for i, group in dataframe.groupby('REF'):
        shifted_values = group.iloc[:,1:-2].shift(1).values
        shifted_dates = group['SurveyFromDiagnosis'].shift(1)
        shifted_dates.iloc[0] = group['WaitingTime'].iloc[0]
        shifted_values[0] = 4
        
        slope_group = (group.iloc[:,1:-2].values - shifted_values) / (pd.concat([(group['SurveyFromDiagnosis'] - shifted_dates)] * (12), axis=1, ignore_index=True))
        slope_group_dict = {}

        for colonna in slope_group:
            slope_group_dict[colonna] = slope_group[colonna].tolist()
            
        slope_group_dict['REF'] = [i]*len(slope_group)
        results = pd.concat([results, pd.DataFrame(slope_group_dict)])
    
    #print(results)
    results.set_index('REF', inplace=True)
    results.rename(columns={0:'ALSFRS_1_slope',
                    1:'ALSFRS_2_slope',
                    2:'ALSFRS_3_slope',
                    3:'ALSFRS_4_slope',
                    4:'ALSFRS_5_slope',
                    5:'ALSFRS_6_slope',
                    6:'ALSFRS_7_slope',
                    7:'ALSFRS_8_slope',
                    8:'ALSFRS_9_slope',
                    9:'ALSFRS_10_slope',
                    10:'ALSFRS_11_slope',
                    11:'ALSFRS_12_slope',}, inplace=True)
    #dictionary_results[column] = results
    
    return results


def convert_ref(dataframe):
    list_ref = []

    try:
        for ref in dataframe['REF']:
            if ref < 10:
                new_ref = 'LIS_000' + str(ref)
            elif ref < 100:
                new_ref = 'LIS_00' + str(ref)
            elif ref < 1000:
                new_ref = 'LIS_0' + str(ref)
            else:
                new_ref = 'LIS_' + str(ref)
            list_ref.append(new_ref)
        return list_ref

    except Exception as error:
        logger.info(f'OSError: {error}')
        return dataframe['REF']


def static_preprocessing(data_torino, data_lisbona):
    data_torino.set_index('REF', inplace=True)
    data_lisbona.set_index('REF', inplace=True)

    logger.info('Dati di Torino: ', data_torino.shape)
    logger.info('Dati di Lisbona: ', data_lisbona.shape)

    data_merged = pd.concat([data_torino, data_lisbona])

    data_merged.dropna(subset=['Status', 'Age_onset (years)', 'NIVFromDiagnosis', 'WaitingTime'], inplace=True)
    data_merged[['NIV', 'PEG']] = data_merged[['NIV', 'PEG']].fillna(value=0)

    data_merged.drop(columns=['TracheostomyFromDiagnosis', 'TARDBP mutation',
                              'FUS mutation', 'SOD1 Mutation '], inplace=True)

    data_merged = pd.get_dummies(data_merged, columns=['UMNvsLMN',
                                                       'Onset', 'ULvsLL', 'DistProx',
                                                       'Side', 'ALS familiar history',
                                                       'smoke', 'Blood hypertension',
                                                       'Diabetes – type I / II', 'Dyslipidemia',
                                                       'Thyroid disorder', 'Autoimmune disorder',
                                                       'Stroke', 'Cardiac disease', 'Primary cancer',
                                                       'C9orf72 repeat-primed PCR result'])

    data_merged = data_merged[data_merged['DateToDeathFromDiagnosis'] >= 0]

    return data_merged


def one_alsfrs(data_merged, data_torino_survey, data_lisbona_survey, data_torino, data_lisbona):
    data_torino_survey.drop(columns=['ALSFRS_TOT', 'Date1'], inplace=True)
    data_lisbona_survey.drop(columns=['Date1'], inplace=True)

    data_torino_survey_min = data_torino_survey.groupby('REF').apply(lambda x: x.loc[abs(x['SurveyFromDiagnosis']).sort_values().index[0],])
    data_lisbona_survey_min = data_lisbona_survey.groupby('REF').apply(lambda x: x.loc[abs(x['SurveyFromDiagnosis']).sort_values().index[0],])

    data_torino_survey_min.set_index('REF', inplace=True)
    data_lisbona_survey_min.set_index('REF', inplace=True)

    data_torino_survey.set_index('REF', inplace=True)
    data_lisbona_survey.set_index('REF', inplace=True)

    data_merged_survey = pd.concat([data_torino_survey_min, data_lisbona_survey_min])
    data_merged = data_merged.join(data_merged_survey)

    data_merged.dropna(subset=['SurveyFromDiagnosis'], inplace=True)

    data_merged = data_merged[(data_merged['SurveyFromDiagnosis'] <= 30) & (data_merged['SurveyFromDiagnosis'] >= -30)]

    data_merged['SurveyFromDeath'] = (data_merged['DateToDeathFromDiagnosis'] - data_merged['SurveyFromDiagnosis'])
    data_merged['Status']=data_merged['Status'].apply(lambda x: False if x == 1.0 else True)

    return data_merged

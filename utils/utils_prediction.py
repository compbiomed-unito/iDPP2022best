import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

COX_pipe = Pipeline([('Scaler', StandardScaler()), ('SimpleImputer', SimpleImputer()),
                        ('COX', CoxnetSurvivalAnalysis())])

RF_pipe = Pipeline([('Scaler', StandardScaler()), ('SimpleImputer', SimpleImputer()),
                        ('RF', RandomSurvivalForest())])

GBSA_pipe = Pipeline([('Scaler', StandardScaler()), ('SimpleImputer', SimpleImputer()),
                        ('GBSA', GradientBoostingSurvivalAnalysis(loss='coxph',
                                                                max_features='sqrt'))])

params_cox = {
    "COX__n_alphas": [50,100,200,300],
    #"COX__alphas": [[a] for a in np.logspace(-4, 0, 20)],
    "COX__l1_ratio": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    #"COX__alpha_min_ratio": [0.001,0.01,0.1]
}

params_rf = {
    "RF__n_estimators": [500],
    "RF__max_depth": [1,2,3,4],
    #"RF__min_samples_split": [2,3,4,10],
    "RF__min_samples_leaf": [5,10,20,50]
}

params_gbsa = {
    "GBSA__learning_rate": [0.9,0.5,0.2,0.1],
    "GBSA__max_depth": [3,4,5],
    "GBSA__n_estimators": [350,400],
    "GBSA__min_samples_leaf": [0.01,0.02,0.05]
}


def bootstrap(grid, X_test, y_test, n_iteration=1000):
    bs_data = pd.DataFrame({
        'estimate': grid.best_estimator_.predict(X_test),
        'event_indicator': y_test['Status'],
        'event_time': y_test['Survival_in_days'],
    })
    bs_cindex = []
    for i in range(n_iteration):
        bs_data_i = bs_data.sample(len(bs_data), replace=True)
        bs_cindex.append(concordance_index_censored(**bs_data_i)[0])
    print(f"Test Score {grid.best_estimator_.score(X_test, y_test)}: {np.quantile(bs_cindex, q=[0.025, 0.975])}: {np.std(bs_cindex)}")


def gridsearchpipeline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=0)

    print('---STARTING COXNET CROSS VALIDATION---')
    grid_cox = GridSearchCV(COX_pipe, params_cox, cv=5, n_jobs=20)
    grid_cox.fit(X_train, y_train)
    print("Training Score: ", grid_cox.score(X_train, y_train))
    bootstrap(grid_cox, X_test, y_test)
    print("Best Parameters: ", grid_cox.best_params_)
    #estimator = grid_cox.best_estimator_
    #dump(estimator, "cox_net.joblib")
    #df = pd.DataFrame(grid_cox.cv_results_)
    #df.to_csv('survey_results/cox_net_death_result.csv')
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, 
                                           grid_cox.best_estimator_.predict(X_test), times=[182, 365, 547, 730])
    print(f'Auc: {auc}, Mean_Auc: {mean_auc}')

    print('---STARTING RANDOM FOREST CROSS VALIDATION---')
    grid_rf = GridSearchCV(RF_pipe, params_rf, cv=5, n_jobs=20)
    grid_rf.fit(X_train, y_train)
    print("Training Score: ", grid_rf.score(X_train, y_train))
    bootstrap(grid_rf, X_test, y_test)
    print("Best Parameters: ", grid_rf.best_params_)
    #estimator = grid_rf.best_estimator_
    #dump(estimator, "random_survival.joblib")
    #df = pd.DataFrame(grid_rf.cv_results_)
    #df.to_csv('survey_results/random_survival_death_result.csv')
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, 
                                           grid_rf.best_estimator_.predict(X_test), times=[182, 365, 547, 730])
    print(f'Auc: {auc}, Mean_Auc: {mean_auc}')

    print('---STARTING GBSA CROSS VALIDATION---')
    grid_gbsa = GridSearchCV(GBSA_pipe, params_gbsa, cv=5, n_jobs=20)
    grid_gbsa.fit(X_train, y_train)
    print("Training Score: ", grid_gbsa.score(X_train, y_train))
    bootstrap(grid_gbsa, X_test, y_test)
    print("Best Parameters: ", grid_gbsa.best_params_)
    #estimator = grid_gbsa.best_estimator_
    #dump(estimator, "gbsa_model.joblib")
    #df = pd.DataFrame(grid_gbsa.cv_results_)
    #df.to_csv('survey_results/gbsa_model_death_result.csv')
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, 
                                           grid_gbsa.best_estimator_.predict(X_test), times=[182, 365, 547, 730])
    print(f'Auc: {auc}, Mean_Auc: {mean_auc}')

def task_death(data_merged):
    y = np.array(data_merged[['Status', 'DateToDeathFromDiagnosis']].apply(tuple, axis=1), 
             dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    X = data_merged[data_merged.columns.difference(['Status', 'DateToDeathFromDiagnosis', 
                                                    'SurveyFromDeath', 
                                                    'NIV', 'PEG', 'Date_NIV', 'Date_PEG', 
                                                    'Date Of Last visit or Death', 
                                                    'NIVFromDiagnosis', 'PEGFromDiagnosis'])]

    gridsearchpipeline(X, y)


def task_niv(data_merged):
    y = np.array(data_merged[['NIV', 'NIVFromDiagnosis']].apply(tuple, axis=1), 
             dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    X = data_merged[data_merged.columns.difference(['Status', 'DateToDeathFromDiagnosis', 'SurveyFromDeath',
                                                    'NIV', 'PEG', 'Date_NIV', 'Date_PEG', 
                                                    'Date Of Last visit or Death', 
                                                    'NIVFromDiagnosis', 'PEGFromDiagnosis'])]

    gridsearchpipeline(X, y)


def task_peg(data_merged):
    y = np.array(data_merged[['PEG', 'PEGFromDiagnosis']].apply(tuple, axis=1), 
             dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    X = data_merged[data_merged.columns.difference(['Status', 'DateToDeathFromDiagnosis', 'SurveyFromDeath', 
                                                    'NIV', 'PEG', 'Date_NIV', 'Date_PEG', 
                                                    'Date Of Last visit or Death', 
                                                    'NIVFromDiagnosis', 'PEGFromDiagnosis'])]

    gridsearchpipeline(X, y)


def task_niv_death(data_merged):
    data_merged['NIVorDeath'] = np.where(data_merged['NIV'] == 1, 1, data_merged['Status'])

    data_merged['NIVorDeathFromDiagnosis'] = np.where(data_merged['NIV'] == 1, 
                                        data_merged['NIVFromDiagnosis'], 
                                        data_merged['DateToDeathFromDiagnosis'])
    
    y = np.array(data_merged[['NIVorDeath', 'NIVorDeathFromDiagnosis']].apply(tuple, axis=1), 
             dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    X = data_merged[data_merged.columns.difference(['Status', 'DateToDeathFromDiagnosis', 'SurveyFromDeath', 
                                                    'NIVorDeath', 'NIVorDeathFromDiagnosis',
                                                    'NIV', 'PEG', 'Date_NIV', 'Date_PEG', 
                                                    'Date Of Last visit or Death', 
                                                    'NIVFromDiagnosis', 'PEGFromDiagnosis'])]

    gridsearchpipeline(X, y)


def task_peg_death(data_merged):
    data_merged['PEGorDeath'] = np.where(data_merged['PEG'] == 1, 1, data_merged['Status'])

    data_merged['PEGorDeathFromDiagnosis'] = np.where(data_merged['PEG'] == 1, 
                                        data_merged['PEGFromDiagnosis'], 
                                        data_merged['DateToDeathFromDiagnosis'])
    
    y = np.array(data_merged[['PEGorDeath', 'PEGorDeathFromDiagnosis']].apply(tuple, axis=1), 
             dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    X = data_merged[data_merged.columns.difference(['Status', 'DateToDeathFromDiagnosis', 'SurveyFromDeath', 
                                                    'PEGorDeath', 'PEGorDeathFromDiagnosis',
                                                    'NIV', 'PEG', 'Date_NIV', 'Date_PEG', 
                                                    'Date Of Last visit or Death', 
                                                    'NIVFromDiagnosis', 'PEGFromDiagnosis'])]

    gridsearchpipeline(X, y)

import pandas as pd

import argparse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow

import warnings
warnings.filterwarnings("ignore")

def parse_arg():
    parser = argparse.ArgumentParser(description='Credit System ML Predictor')
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='The number of trees in the forest.'
    )
    parser.add_argument(
        '--criterion',
        type=str,
        default='entropy',
        help='''
        The function to measure the quality of a split. 
        Supported criteria are “gini” for the Gini impurity 
        and “log_loss” and “entropy” both for the Shannon 
        information gain;
    '''
    )

    parser.add_argument(
        '--max-depth',
        type=int,
        default=5,
        help='''
        The maximum depth of the tree. If None, then nodes are 
        expanded until all leaves are pure or until all leaves 
        contain less than min_samples_split samples.  
    '''
    )
    return parser.parse_args()

data = pd.read_csv('data/preprocessed/preprocessed_data_combined.csv')

features = ['personal_loan', 'securities_account', 'cd_account', 'online',
            'cat__age_bracket_name_Baby boomers', 'cat__age_bracket_name_Generation X',
            'cat__age_bracket_name_Generation Z', 'cat__age_bracket_name_Millennials',
            'cat__education_ensino_medio', 'cat__education_ensino_superior',
            'cat__education_pos_graduacao']

X, y = data[features], data['credit_card']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

rus = RandomUnderSampler(random_state=42)

X_res, y_res = rus.fit_resample(X_train, y_train)

def main():
    args = parse_arg()
    rf_params = {'n_estimators': args.n_estimators,
                 'criterion': args.criterion,
                 'max_depth': args.max_depth}

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('test-model')

    with mlflow.start_run(run_name='RandomForestClassifier-CreditApproval'):
        mlflow.sklearn.autolog()
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_res, y_res)
        y_pred = rf_model.predict(X_test)

        acc = accuracy_score(y_pred, y_test)
        precision = precision_score(y_pred, y_test)
        recall = recall_score(y_pred, y_test)
        print('Acurácia: {}'.format(acc))
        print('Precisão: {}'.format(precision))
        print('Revocação: {}'.format(recall))
        mlflow.log_metric('Acurácia', acc)
        mlflow.log_metric('Revocação', recall)
        mlflow.log_metric('Precisão', precision)

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import dvc.api
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, roc_auc_score, auc, confusion_matrix
import mlflow
import mlflow.sklearn

import os
import warnings
import sys
import pathlib
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

PATH=pathlib.Path(__file__).parent
DATA_PATH=PATH.joinpath("./data").resolve()

path = DATA_PATH.joinpath("data.csv")
repo = 'C:\Users\J\Desktop\10_Academy\week2\final task\abtest-mlops\'
version = 'v2'

data_url = dvc.api.get_url(path=path,
                           repo=repo,
                           rev=version)

mlflow.set_experiment("abtest")

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    return accuracy, precision, recall

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(40)

    data = pd.read_csv(data_url, sep=',')

    data_1 = data.drop(['no','date','auction_id'], axis=1) 

    label_encoder = LabelEncoder()
    data_1['experiment'] = label_encoder.fit_transform(data_1['experiment'])
    data_1['device_make'] = label_encoder.fit_transform(data_1['device_make'])
    data_1['browser'] = label_encoder.fit_transform(data_1['browser'])
    data_1= data_1.rename(columns={"yes":"brand_awareness"})

    X = data_1.drop('brand_awareness', axis = 1) # features
    y = data_1['brand_awareness']# target

    X_train, X_remain, y_train, y_remain = train_test_split(X, y, train_size= 0.7)
    X_validation, X_test,y_validation, y_test = train_test_split(X_remain, y_remain, test_size= 0.33)

    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_validation)
    log_result = log_model.score(X_test, y_test)
    print("The accuracy: %.2f%%" % (log_result*100.0))

    (accuracy, precision, recall) = eval_metrics(y_validation, y_pred)
    # Report training set score
    train_score = log_model.score(X_train, y_train) * 100
    # Report test set score
    test_score = log_model.score(X_test, y_test) * 100

    # Write scores to a file
    with open("metrics.txt", 'w') as outfile:
            outfile.write("Training accuracy explained: %2.1f%%\n" % train_score)
            outfile.write("Test accuracy explained: %2.1f%%\n" % test_score)


    Kfold = KFold(n_splits = 5, random_state = 30)
    model_kfold = LogisticRegression()
    kfold_result = cross_val_score(model_kfold, X_train, y_train, cv = Kfold )
    print("The accuracy : %.2f%%" % (kfold_result.mean()*100.0))

    grid_search = GridSearchCV(log_model, {"C":np.logspace(-3,3,15), "penalty":["l1","l2"]}, cv = 5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_valid__pred = grid_search.predict(X_validation)
    grid_search_score = grid_search.score(X_validation, y_validation)
    print("The accuracy : %.2f%%" % (grid_search_score.mean()*100.0))

    # Log mlflow attributes for mlflow UI

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.sklearn.log_model(log_model, "model")
    
    log_confusionM = confusion_matrix(y_validation, y_valid__pred)
    print('precision :', precision_score(y_validation, y_valid__pred, average='weighted'))
    print('recall: ', recall_score(y_validation, y_valid__pred,  average='weighted'))
    print(classification_report(y_validation, y_valid__pred))

    sns.set(style = 'darkgrid')
    heatmap = sns.heatmap(log_confusionM/ np.sum(log_confusionM), vmin=-1, vmax=1, annot = True, annot_kws={"size": 10}, cmap="BrBG")
    heatmap.set_title('Confusion matrix Heatmap', fontdict={'fontsize':18}, pad=12);
    plt.ylabel('true label', fontdict={'fontsize':13})
    plt.xlabel('Predicted label', fontdict={'fontsize':13})
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    
    
    loss= log_loss(y_validation, y_valid__pred)
    print("The loss : %.2f%%" % (loss))

    feature_importance_dict = dict(zip(X_train.columns, log_model.coef_[0]))
    feature_imp_dict = pd.DataFrame.from_dict(feature_importance_dict, orient='index')
    feature_imp_dict.rename(columns = {0:'Feature importance'}, inplace = True)
    feature_imp_dict.sort_values(by=['Feature importance'], ascending=False)

    sns.set(style = 'darkgrid')
    sns.barplot(x= feature_imp_dict.index,y="Feature importance",data = feature_imp_dict)
    plt.savefig("feature_importance.png", dpi=120)
    mlflow.log_artifact("feature_importance.png")
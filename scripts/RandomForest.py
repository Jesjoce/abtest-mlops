import pandas as pd
import numpy as np
import dvc.api
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

    max_depth=None
    max_features='auto'
    max_samples=None
    n_estimators=100

    clf = RandomForestClassifier( n_estimators = 50, 
                                criterion='gini', 
                                max_depth=5, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, 
                                max_features ='auto', 
                                max_leaf_nodes=None, 
                                min_impurity_decrease=0.0, 
                                min_impurity_split = None, 
                                bootstrap = True, 
                                oob_score=False, 
                                n_jobs =1, 
                                class_weight = 'balanced', 
                                warm_start = False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_validation)
    (accuracy, precision, recall) = eval_metrics(y_validation, y_pred)
    # Report training set score
    train_score = clf.score(X_train, y_train) * 100
    # Report test set score
    test_score = clf.score(X_test, y_test) * 100

    # Write scores to a file
    with open("metrics.txt", 'w') as outfile:
            outfile.write("Training accuracy explained: %2.1f%%\n" % train_score)
            outfile.write("Test accuracy explained: %2.1f%%\n" % test_score)

    clf_result = clf.score(X_validation, y_validation)
    print("The accuracy of clf: %.2f%%" % (clf_result*100.0))

    cv = KFold(n_splits = 5, random_state = 50, shuffle = True)
    Kfold_clf_results = cross_val_score(clf, X_validation, y_validation, cv = cv )
    print("The accuracy of clf using KFold : %.2f%%" % (Kfold_clf_results.mean()*100.0))

    param_grid = {'bootstrap': [True],
                'max_depth': [5, 6, 7, 8, 9],
                'max_features': [2,3],
                'min_samples_leaf':[3,4,5],
                'min_samples_split': [8,10,12],
                'n_estimators':[100,200,300,1000] 
    }
    grid_search_clf = GridSearchCV(clf, param_grid = param_grid, cv =5, n_jobs = -1, verbose=2)
    grid_search_clf.fit(X_train, y_train)
    y_pred = grid_search_clf.predict(X_validation)

    grid_search_score = grid_search_clf.score(X_validation, y_validation)
    print("The accuracy : %.2f%%" % (grid_search_score.mean()*100.0))

    #Log mlflow attributes for mlflow UI
    mlflow.log_param("max_depth ", max_depth )
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("max_samples", max_samples)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.sklearn.log_model(clf, "model")


    clf_confusionM = confusion_matrix(y_validation, y_pred)
    print('precision :', precision_score(y_validation, y_pred, average='weighted'))
    print('recall: ', recall_score(y_validation, y_pred,  average='weighted'))
    print(classification_report(y_validation, y_pred))

    sns.set(style = 'darkgrid')
    heatmap = sns.heatmap(clf_confusionM/ np.sum(clf_confusionM), vmin=-1, vmax=1, annot = True, annot_kws={"size": 10}, cmap="BrBG")
    heatmap.set_title('Confusion matrix Heatmap', fontdict={'fontsize':18}, pad=12);
    plt.ylabel('true label', fontdict={'fontsize':13})
    plt.xlabel('Predicted label', fontdict={'fontsize':13})
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.show()

    loss= log_loss(y_validation, y_pred)
    print("The loss : %.2f%%" % (loss))

    clf_feature_importance_dict = dict(zip(X_train.columns, clf.feature_importances_))
    clf_feature_imp_dict = pd.DataFrame.from_dict(clf_feature_importance_dict, orient='index')
    clf_feature_imp_dict.rename(columns = {0:'Feature importance'}, inplace = True)
    clf_feature_imp_dict.sort_values(by=['Feature importance'], ascending=False)

    sns.set(style = 'darkgrid')
    sns.barplot(x= clf_feature_imp_dict.index,y="Feature importance",data = clf_feature_imp_dict)
    plt.savefig("feature_importance.png", dpi=120)
    plt.show()

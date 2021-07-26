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


XGB_model = XGBClassifier()
XGB_model.fit(X_train, y_train)
XBG_y_pred = XGB_model.predict(X_validation)
accuracy = accuracy_score(y_validation, XBG_y_pred)
print("The Accuracy score of XGBoost : %.2f%%" % (accuracy * 100.0)

Kfold = KFold(n_splits = 5, random_state = 30)
XGB_kfold_result = cross_val_score(XGB_model, X_train, y_train, cv = Kfold )
print("The accuracy of XGBoost with KFold: %.2f%%" % (XGB_kfold_result.mean()*100.0))

param_grid = {
    'n_estimators': [40, 70, 100],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [5,10,15],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'subsample': [0.7, 0.8, 0.9]
}
XGBoost_grid_search = GridSearchCV(XGB_model, param_grid = param_grid, cv = 5, n_jobs=-1)
XGBoost_grid_search.fit(X_train, y_train)
y_val__pred = XGBoost_grid_search.predict(X_validation)
XGBoost_grid_search_score = XGBoost_grid_search.score(X_validation, y_validation)
print("The accuracy of XGBoost with searchgridCV: %.2f%%" % (XGBoost_grid_search_score.mean()*100.0))

XGBoost_confusionM = confusion_matrix(y_validation, y_val__pred)
print('precision :', precision_score(y_validation, y_val__pred, average='weighted'))
print('recall: ', recall_score(y_validation, y_val__pred,  average='weighted'))
print(classification_report(y_validation, y_val__pred))

sns.set(style = 'darkgrid')
heatmap = sns.heatmap(XGBoost_confusionM/ np.sum(XGBoost_confusionM), vmin=-1, vmax=1, annot = True, annot_kws={"size": 10}, cmap="BrBG")
heatmap.set_title('Confusion matrix Heatmap', fontdict={'fontsize':18}, pad=12);
plt.ylabel('true label', fontdict={'fontsize':13})
plt.xlabel('Predicted label', fontdict={'fontsize':13})
loss= log_loss(y_validation, y_val__pred)
print("The loss : %.2f%%" % (loss))

XGB_feature_importance_dict = dict(zip(X_train.columns, log_model.coef_[0]))
XGB_feature_imp_dict = pd.DataFrame.from_dict(XGB_feature_importance_dict, orient='index')
XGB_feature_imp_dict.rename(columns = {0:'Feature importance'}, inplace = True)
XGB_feature_imp_dict.sort_values(by=['Feature importance'], ascending=False)

sns.set(style = 'darkgrid')
sns.barplot(x= feature_imp_dict.index,y="Feature importance",data = XGB_feature_imp_dict)
plt.show()




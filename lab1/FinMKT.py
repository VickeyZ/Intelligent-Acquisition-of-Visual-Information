import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# some usable model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')


def age(dataframe):
    dataframe.loc[dataframe['age'] <= 32, 'age'] = 1
    dataframe.loc[(dataframe['age'] > 32) & (dataframe['age'] <= 47), 'age'] = 2
    dataframe.loc[(dataframe['age'] > 47) & (dataframe['age'] <= 70), 'age'] = 3
    dataframe.loc[(dataframe['age'] > 70) & (dataframe['age'] <= 98), 'age'] = 4

    return dataframe

def duration(data):

    data.loc[data['duration'] <= 102, 'duration'] = 1
    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration']    = 2
    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration']   = 3
    data.loc[(data['duration'] > 319) & (data['duration'] <= 644.5), 'duration'] = 4
    data.loc[data['duration']  > 644.5, 'duration'] = 5

    return data

def data_preprocess(bank):

    y = pd.get_dummies(bank['y'], columns=['y'], prefix=['y'], drop_first=True)
    bank_client = bank.iloc[:, 0:7]
    bank_related = bank.iloc[:, 7:11]
    bank_se = bank.loc[:, ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
    bank_o = bank.loc[:, ['campaign', 'pdays', 'previous', 'poutcome']]


    labelencoder_X = LabelEncoder()
    bank_client['job'] = labelencoder_X.fit_transform(bank_client['job'])
    bank_client['marital'] = labelencoder_X.fit_transform(bank_client['marital'])
    bank_client['education'] = labelencoder_X.fit_transform(bank_client['education'])
    bank_client['default'] = labelencoder_X.fit_transform(bank_client['default'])
    bank_client['housing'] = labelencoder_X.fit_transform(bank_client['housing'])
    bank_client['loan'] = labelencoder_X.fit_transform(bank_client['loan'])
    bank_client = age(bank_client)

    labelencoder_X = LabelEncoder()
    bank_related['contact'] = labelencoder_X.fit_transform(bank_related['contact'])
    bank_related['month'] = labelencoder_X.fit_transform(bank_related['month'])
    bank_related['day_of_week'] = labelencoder_X.fit_transform(bank_related['day_of_week'])
    bank_related = duration(bank_related)

    bank_o['poutcome'].replace(['nonexistent', 'failure', 'success'], [1, 2, 3], inplace=True)

    bank_final = pd.concat([bank_client, bank_related, bank_se, bank_o], axis=1)
    bank_final = bank_final[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                             'contact', 'month', 'day_of_week', 'duration', 'emp.var.rate', 'cons.price.idx',
                             'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays', 'previous', 'poutcome']]


    return bank_final, y

def LRpredict(X_train, X_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    y_pred = logmodel.predict(X_test)
    LOGCV = (cross_val_score(logmodel, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())
    # your code here end

    return y_pred,LOGCV

def KNNpredict(X_train, X_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    knn = KNeighborsClassifier(n_neighbors=22)
    knn.fit(X_train, y_train)
    knnpred = knn.predict(X_test)
    KNNCV = (cross_val_score(knn, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())    # your code here end

    return knnpred,KNNCV

def SVCpredict(X_train, X_test, y_train):

    svc = SVC(kernel='sigmoid')
    svc.fit(X_train, y_train)
    svcpred = svc.predict(X_test)

    SVCCV = (cross_val_score(svc, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())

    return svcpred , SVCCV

def DTreepredict(X_train, X_test, y_train):

    dtree = DecisionTreeClassifier(criterion='gini')  # criterion = entopy, gini
    dtree.fit(X_train, y_train)
    dtreepred = dtree.predict(X_test)

    DTREECV = (cross_val_score(dtree, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())

    return dtreepred, DTREECV

def RFCpredict(X_train, X_test, y_train):

    rfc = RandomForestClassifier(n_estimators=200)  # criterion = entopy,gini
    rfc.fit(X_train, y_train)
    rfcpred = rfc.predict(X_test)

    RFCCV = (cross_val_score(rfc, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())

    return rfcpred, RFCCV

def Gausspredict(X_train, X_test, y_train):

    gaussiannb = GaussianNB()
    gaussiannb.fit(X_train, y_train)
    gaussiannbpred = gaussiannb.predict(X_test)
    probs = gaussiannb.predict(X_test)

    GAUSIAN = (cross_val_score(gaussiannb, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())

    return probs, GAUSIAN



def GBKpredict(X_train, X_test, y_train):

    gbk = GradientBoostingClassifier()
    gbk.fit(X_train, y_train)
    gbkpred = gbk.predict(X_test)
    print(confusion_matrix(y_test, gbkpred))
    print(round(accuracy_score(y_test, gbkpred), 2) * 100)
    GBKCV = (cross_val_score(gbk, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())

    return gbkpred, GBKCV



def split_data(data):
    y = data.y
    x = data.loc[:, data.columns != 'y']
    x = data_preprocess(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test

def print_result(y_test, y_pred):
    report = confusion_matrix(y_test, y_pred)
    precision = report[1][1] / (report[:, 1].sum())
    recall = report[1][1] / (report[1].sum())
    print('model precision:' + str(precision)[:4] + ' recall:' + str(recall)[:4])

if __name__ == '__main__':
    #bank = pd.read_csv('bank-additional-full.csv', sep=';')

    #bank_final, y = data_preprocess(bank)
    #k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

    #X_train, X_test, y_train, y_test = train_test_split(bank_final, y, test_size=0.1942313295, random_state=101)

    #sc_X = StandardScaler()
    #X_train = sc_X.fit_transform(X_train)
    #X_test = sc_X.transform(X_test)

    #y_pred,a = KNNpredict(X_train, X_test, y_train)

    #print('KNN Reports\n', classification_report(y_test, y_pred))
    #print('Accuracy:',accuracy_score(y_test, y_pred))
    #print_result(y_test, y_pred)

    data = pd.read_csv("final_Data.csv")
    list = data.values.tolist()
    list_ndarray = np.array(list)
    x_train = list_ndarray[:,0:2]
    x_train = x_train.tolist()
    y_train = list_ndarray[:,2:]
    y_train = y_train.tolist()

    list_t = [y_train[i][0] for i in range(len(y_train))]
    y_train = list_t
    model=LinearRegression()
    model.fit(x_train,y_train)
    w = model.coef_
    b = model.intercept_  # 得到bias值
    print(len(w))  # 输出参数数目
    print([round(i, 5) for i in w])  # 输出w列表，保留5位小数
    print(b)  # 输出bias

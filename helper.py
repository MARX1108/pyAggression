from sklearn.model_selection import train_test_split
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from imblearn.over_sampling import SMOTE

import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import (confusion_matrix,
                             accuracy_score, classification_report)

import pandas as pd
import pickle
base = importr('base')
ROSE = importr('ROSE')


def preprosessing(columns, seed=3, size=2000, original_file="new-aggr-all-processed.csv", orginal=False, split=False, train_size=0.75, random_state=0):
    df = pd.read_csv(f"./data/{original_file}")

    assert 'y' in df

    for index, item in enumerate(columns):
        with open(f'./data/columns/{item}', 'rb') as filehandle:
            item = pickle.load(filehandle)
        if(index == 0):
            temp = pd.concat([df[item], df['y']], axis=1)
        else:
            temp = pd.concat([df[item], temp], axis=1)

    X = temp.drop('y', axis=1)
    y = temp['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state, train_size=train_size)

    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    print(y.value_counts())
    print(y_train.value_counts())
    print(y_test.value_counts())

    # print(X_train.shape)
    # print(X_test.shape)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    if not orginal:
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_from_pd_df = ro.conversion.py2rpy(train)

        ro.globalenv["df"] = r_from_pd_df
        ro.r(f'''
                if('sex' %in% colnames(df))
                df$sex <- as.factor(df$sex)
                data.rose <- ROSE(y~., data=df, seed={seed},
                        N={size})$data
    ''')

        with localconverter(ro.default_converter + pandas2ri.converter):
            pd_from_r_df = ro.conversion.rpy2py(ro.globalenv["data.rose"])

        assert 'y' in pd_from_r_df
        if split:
            return pd_from_r_df, test
        else:
            return pd_from_r_df

    else:
        return temp


def logit(col, rose_size=5000, train_size=0.75, split_random_state=3):
    # col = ["neighbor"]
    train, test = preprosessing(
        col, size=rose_size, seed=3, split=True, train_size=train_size, random_state=split_random_state)
    if 'sex' in col:
        train = pd.get_dummies(train, columns=['sex'], drop_first=True)
        test = pd.get_dummies(test, columns=['sex'], drop_first=True)

    ytrain = train[['y']]
    Xtrain = train.drop('y', axis=1)
    Xtrain = sm.add_constant(Xtrain)
    Xtest = test.drop('y', axis=1)
    Xtest = sm.add_constant(Xtest)

    log_reg = sm.Logit(ytrain, Xtrain).fit()
    yhat = log_reg.predict(Xtest)
    train_yhat = log_reg.predict(Xtrain)

    train_prediction = list(map(round, train_yhat))
    prediction = list(map(round, yhat))
    # print(log_reg.summary())

    # confusion matrix
    cm = confusion_matrix(test[['y']], prediction)
    # print ("Confusion Matrix : \n", cm)
    # accuracy score of the model

    report = classification_report(test[['y']], prediction, target_names=[
                                   'Low', 'High'], output_dict=True)

    # accuracy score of the model
    train_accuracy = accuracy_score(train[['y']], train_prediction)
    test_accuracy = accuracy_score(test[['y']], prediction)
    # print('Train accuracy = ', train_accuracy)
    # print('Test accuracy = ', test_accuracy)
    return train_accuracy, test_accuracy, report

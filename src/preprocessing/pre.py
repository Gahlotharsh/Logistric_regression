import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\harsh\PycharmProjects\Breast_Cancer_prediction\src\data\breast-cancer.csv")


def split(data_set):
    x = data_set.drop('diagnosis',axis = 1)
    y = data_set['diagnosis']
    x_train, x_test, y_train, y_test = (train_test_split(x, y, test_size=.2, random_state=2529))
    return x_train, x_test, y_train, y_test


def labelEncode(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test


def scaler(x_train, x_test):
    scal = StandardScaler()
    scal.fit(x_train)
    x_train = scal.transform(x_train)
    x_test = scal.transform(x_test)
    return x_train, x_test


def preprocessing_runner():
    x_train, x_test, y_train, y_test = split(df)
    y_train, y_test = labelEncode(y_train, y_test)
    x_train, x_test = scaler(x_train, x_test)
    # print(x_test)
    return x_train, x_test, y_train, y_test





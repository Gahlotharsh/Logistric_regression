from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def model(x_train, y_train):
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    return lr


def predict(lr, x_test):
    y_predict = lr.predict(x_test)
    return y_predict


def accuracy_model(y_test, y_predict):
    accuracy = accuracy_score(y_test, y_predict)
    return accuracy


def model_runner(x_train, y_train, x_test, y_test):
    lr = model(x_train, y_train)
    y_predict = predict(lr, x_test)
    # print(y_predict)
    accuracy = accuracy_model(y_test, y_predict)
    print(accuracy)

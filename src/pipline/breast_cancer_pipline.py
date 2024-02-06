from src.preprocessing.pre import preprocessing_runner
from src.brest_cancer_prediction_model.linear_regression_model import model_runner


def pipline():
    x_train, x_test, y_train, y_test = preprocessing_runner()
    model_runner(x_train, y_train, x_test, y_test)


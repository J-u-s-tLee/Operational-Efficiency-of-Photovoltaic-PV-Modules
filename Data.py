import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path, keep_default_na=False)
    return data

def split_labels(data):
    expected_efficiency = data[["expected_efficiency"]]
    efficiency_level = data[["efficiency_level"]]
    return expected_efficiency, efficiency_level

def split_data(data, column_1, column_2, test_size=0.3, random_state=42, model_type="Linear Regression"):
    X = data.drop(columns=[column_1, column_2])
    y = data[[column_1, column_2]]

    if model_type == "Linear Regression":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        y_train_1, y_train_2 = split_labels(y_train)
        y_test_1, y_test_2 = split_labels(y_test)
        return X_train, X_test, y_train_1, y_test_1
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)
        y_train_1, y_train_2 = split_labels(y_train)
        y_validation_1, y_validation_2 = split_labels(y_validation)
        y_test_1, y_test_2 = split_labels(y_test)
        return X_train, X_validation, X_test, y_train_1, y_validation_1, y_test_1,  y_train_2, y_validation_2, y_test_2

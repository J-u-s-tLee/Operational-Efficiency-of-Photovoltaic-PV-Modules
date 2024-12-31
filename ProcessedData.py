import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

def load_data(file_path):
    data = pd.read_csv(file_path, keep_default_na=False)
    return data

def map_efficiency_level_to_numeric(data):
    efficiency_mapping = {
        'extremely_bad': 0,
        'bad': 1,
        'moderate': 2,
        'good': 3,
    }
    
    data['efficiency_level'] = data['efficiency_level'].map(efficiency_mapping)
    
    return data


def split_labels(data):
    expected_efficiency = data[["expected_efficiency"]]
    efficiency_level = data[["efficiency_level"]]
    return expected_efficiency, efficiency_level

def split_data(data, column_1, column_2, test_size=0.3, random_state=42):
    X = data.drop(columns=[column_1, column_2])
    y = data[[column_1, column_2]]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)
    y_train_1, y_train_2 = split_labels(y_train)
    y_validation_1, y_validation_2 = split_labels(y_validation)
    y_test_1, y_test_2 = split_labels(y_test)
    return X_train, X_validation, X_test, y_train_1, y_validation_1, y_test_1, y_train_2, y_validation_2, y_test_2

def encode_categorical(X_train, X_validation, X_test, categorical_cols):
    encoder = OneHotEncoder(sparse_output=False)
    X_train_categorical = encoder.fit_transform(X_train[categorical_cols])
    X_validation_categorical = encoder.transform(X_validation[categorical_cols])
    X_test_categorical = encoder.transform(X_test[categorical_cols])
    return X_train_categorical, X_validation_categorical, X_test_categorical, encoder

def normalize_numerical(X_train, X_validation, X_test, numerical_cols):

    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train[numerical_cols])
    X_validation_normalized = scaler.transform(X_validation[numerical_cols])
    X_test_normalized = scaler.transform(X_test[numerical_cols])

    X_train_normalized = pd.DataFrame(X_train_normalized, columns=numerical_cols, index=X_train.index)
    X_validation_normalized = pd.DataFrame(X_validation_normalized, columns=numerical_cols, index=X_validation.index)
    X_test_normalized = pd.DataFrame(X_test_normalized, columns=numerical_cols, index=X_test.index)
    return X_train_normalized, X_validation_normalized, X_test_normalized, scaler

def standardize_numerical(X_train, X_validation, X_test, numerical_cols):
    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train[numerical_cols])
    X_validation_standardized = scaler.transform(X_validation[numerical_cols])
    X_test_standardized = scaler.transform(X_test[numerical_cols])

    X_train_standardized = pd.DataFrame(X_train_standardized, columns=numerical_cols, index=X_train.index)
    X_validation_standardized = pd.DataFrame(X_validation_standardized, columns=numerical_cols, index=X_validation.index)
    X_test_standardized = pd.DataFrame(X_test_standardized, columns=numerical_cols, index=X_test.index)
    return X_train_standardized, X_validation_standardized, X_test_standardized, scaler

def combine_processed_data(X_train, X_validation, X_test, 
                           X_train_categorical, X_validation_categorical, X_test_categorical, encoder, categorical_cols,
                           X_train_numerical_standardized, X_validation_numerical_standardized, X_test_numerical_standardized):

    categorical_feature_names = encoder.get_feature_names_out(categorical_cols)

    X_train_combined = pd.concat([ 
        pd.DataFrame(X_train_categorical, index=X_train.index, columns=categorical_feature_names),
        X_train_numerical_standardized
    ], axis=1)

    
    X_validation_combined = pd.concat([ 
        pd.DataFrame(X_validation_categorical, index=X_validation.index, columns=categorical_feature_names),
        X_validation_numerical_standardized
    ], axis=1)
    
    X_test_combined = pd.concat([ 
        pd.DataFrame(X_test_categorical, index=X_test.index, columns=categorical_feature_names),
        X_test_numerical_standardized
    ], axis=1)
    
    return X_train_combined, X_validation_combined, X_test_combined

def processed_data(X_train, X_validation, X_test, categorical_cols):
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()  
    
    X_train_categorical, X_validation_categorical, X_test_categorical, encoder = encode_categorical(X_train, X_validation, X_test, categorical_cols)
    
    X_train_normalized, X_validation_normalized, X_test_normalized, minmax_scaler = normalize_numerical(X_train, X_validation, X_test, numerical_cols)
    
    X_train_standardized, X_validation_standardized, X_test_standardized, standard_scaler = standardize_numerical(X_train_normalized, X_validation_normalized, X_test_normalized, numerical_cols)
    
    X_train_processed, X_validation_processed, X_test_processed = combine_processed_data(
        X_train, X_validation, X_test,
        X_train_categorical, X_validation_categorical, X_test_categorical, encoder, categorical_cols,
        X_train_standardized, X_validation_standardized, X_test_standardized
    )
    
    return X_train_processed, X_validation_processed, X_test_processed

import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
from sklearn.preprocessing import OneHotEncoder
from ProcessedData import load_data, map_efficiency_level_to_numeric, split_data, processed_data
from PCA import apply_pca
from after_PCA import aggregate_importance_by_original_features
from model import SharedFeedForwardNN
from train import train_model 
from test import test_model

class Parameters:
    def __init__(self):
        self.file_path = 'C:\\Users\\leono\\OneDrive\\Ambiente de Trabalho\\Bioengenharia\\DACO\\Project\\Dataset\\data.csv' # Change to dataset path 
        self.project_dir = os.path.abspath(os.path.join(self.file_path, "..", "..")) 
        self.save_dir = os.path.join(self.project_dir, 'Results')
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.categorical_cols = ['pv_module_type', 'hotspot', 'birddrop', 'soiling', 'junction_box']
        self.numerical_cols = ['affected_area', 'temperature', 'irradiance', 'Voc', 'Isc']
        self.num_epochs = 50
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.alfa = 0.5
        self.beta = 0.5
        self.num_classes = 4
        self.hidden_dim = 32
        self.dropout_rate = 0.3
        self.patience = 10
        self.n_components = 2

def main():

    params = Parameters()

    data = load_data(params.file_path)
    data = map_efficiency_level_to_numeric(data)

    X_train, X_val, X_test, Y_train_1, Y_val_1, Y_test_1, Y_train_2, Y_val_2, Y_test_2 = split_data(data, 'expected_efficiency', 'efficiency_level')

    X_train_processed, X_val_processed, X_test_processed = processed_data(X_train, X_val, X_test, params.categorical_cols)

    input_dim = X_train_processed.shape[1]

    model = SharedFeedForwardNN(
        input_dim=input_dim, 
        hidden_dim=params.hidden_dim, 
        num_classes=params.num_classes, 
        dropout_rate=params.dropout_rate
    )

    X_train_tensor = torch.tensor(X_train_processed.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(
        pd.concat([Y_train_1, Y_train_2], axis=1).values, dtype=torch.float32
    )

    X_val_tensor = torch.tensor(X_val_processed.values, dtype=torch.float32)
    Y_val_tensor = torch.tensor(
        pd.concat([Y_val_1, Y_val_2], axis=1).values, dtype=torch.float32
    )

    X_test_tensor = torch.tensor(X_test_processed.values, dtype=torch.float32)
    Y_test_tensor = torch.tensor(
        pd.concat([Y_test_1, Y_test_2], axis=1).values, dtype=torch.float32
    )

    train_loader = DataLoader(torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor), batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor), batch_size=32, shuffle=True, num_workers=8)
    test_loader = DataLoader(torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor), batch_size=32, shuffle=False, num_workers=8)
    
    train_model(
        model,
        train_loader,
        val_loader, 
        num_epochs=params.num_epochs, 
        learning_rate=params.learning_rate, 
        weight_decay=params.weight_decay, 
        alfa=params.alfa, 
        beta=params.beta, 
        patience=params.patience, 
        save_dir=params.save_dir,
        device=params.device
    )

    test_model(
        model,
        test_loader,
        num_classes=params.num_classes,
        save_dir=params.save_dir,
        load_dir=params.save_dir)

    _, _, importance_df_train = apply_pca(X_train_processed, params.n_components)
    
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(X_train[params.categorical_cols])

    consolidated_df = aggregate_importance_by_original_features(
        importance_df_train,  
        params.categorical_cols,    
        params.numerical_cols,       
        X_train_processed,
        encoder,             
        params.save_dir             
    )

    # Training with the two most important data

    top_features = consolidated_df.head(2)['Feature'].tolist()

    categorical_in_top_features = [col for col in top_features if col in params.categorical_cols]

    if categorical_in_top_features:

        params.categorical_cols = categorical_in_top_features

    else:
        
        params.categorical_cols = []

    X_train_top_features = X_train[top_features]
    X_val_top_features = X_val[top_features]
    X_test_top_features = X_test[top_features]


    X_train_top_features_processed, X_val_top_features_processed, X_test_top_features_processed = processed_data(X_train_top_features, X_val_top_features, X_test_top_features, params.categorical_cols)

    input_dim = X_train_top_features_processed.shape[1]

    model = SharedFeedForwardNN(
        input_dim=input_dim, 
        hidden_dim=params.hidden_dim, 
        num_classes=params.num_classes, 
        dropout_rate=params.dropout_rate
    )

    X_train_tensor = torch.tensor(X_train_top_features_processed.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(
        pd.concat([Y_train_1, Y_train_2], axis=1).values, dtype=torch.float32
    )

    X_val_tensor = torch.tensor(X_val_top_features_processed.values, dtype=torch.float32)
    Y_val_tensor = torch.tensor(
        pd.concat([Y_val_1, Y_val_2], axis=1).values, dtype=torch.float32
    )

    X_test_tensor = torch.tensor(X_test_top_features_processed.values, dtype=torch.float32)
    Y_test_tensor = torch.tensor(
        pd.concat([Y_test_1, Y_test_2], axis=1).values, dtype=torch.float32
    )

    train_loader = DataLoader(torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor), batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor), batch_size=32, shuffle=True, num_workers=8)
    test_loader = DataLoader(torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor), batch_size=32, shuffle=False, num_workers=8)

    params.save_dir = os.path.join(params.project_dir, 'Results2')
    os.makedirs(params.save_dir, exist_ok=True)

    print('\nModel performance for two principal features as input:')
    train_model(
        model,
        train_loader,
        val_loader, 
        num_epochs=params.num_epochs, 
        learning_rate=params.learning_rate, 
        weight_decay=params.weight_decay, 
        alfa=params.alfa, 
        beta=params.beta, 
        patience=params.patience, 
        save_dir=params.save_dir,
        device=params.device
    )

    test_model(
        model,
        test_loader,
        num_classes=params.num_classes,
        save_dir=params.save_dir,
        load_dir=params.save_dir)



if __name__ == "__main__":
    main()

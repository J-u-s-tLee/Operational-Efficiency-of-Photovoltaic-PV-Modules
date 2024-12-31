import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from ProcessedData import load_data, map_efficiency_level_to_numeric, split_data, processed_data
from PCA import apply_pca
from after_PCA import aggregate_importance_by_original_features
from model import SharedFeedForwardNN
from train import train_model 
from test import test_model

def main():
    file_path = 'C:\\Users\\leono\\OneDrive\\Ambiente de Trabalho\\Bioengenharia\\DACO\\Project\\Dataset\\data.csv'
    save_dir = 'C:\\Users\\leono\\OneDrive\\Ambiente de Trabalho\\Bioengenharia\\DACO\\Project\\Results'

    data = load_data(file_path)
    data = map_efficiency_level_to_numeric(data)

    categorical_cols = ['pv_module_type', 'hotspot', 'birddrop', 'soiling', 'junction_box']
    numerical_cols =['affected_area', 'temperature', 'irradiance', 'Voc', 'Isc']

    X_train, X_val, X_test, Y_train_1, Y_val_1, Y_test_1, Y_train_2, Y_val_2, Y_test_2 = split_data(data, 'expected_efficiency', 'efficiency_level')

    X_train_processed, X_val_processed, X_test_processed = processed_data(X_train, X_val, X_test, categorical_cols)

    num_epochs = 20
    learning_rate = 1e-3
    weight_decay = 1e-5
    alfa = 0.5  
    beta = 0.5
    num_classes = 4
    hidden_dim = 64

    input_dim = X_train_processed.shape[1] 
    model = SharedFeedForwardNN(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes) 

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
    
    #train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay, alfa, beta, save_dir=save_dir)

    #test_model(model, test_loader, num_classes=num_classes, save_dir=save_dir)

    n_components = 2
    _, _, importance_df_train = apply_pca(X_train_processed, n_components)
    
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(X_train[categorical_cols])

    consolidated_df = aggregate_importance_by_original_features(
        importance_df_train,  
        categorical_cols,    
        numerical_cols,       
        X_train_processed,   
        encoder,             
        save_dir             
    )


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

from data import load_data, split_data, preprocess_data

if __name__ == "__main__":
    #Upload and preprocessement of data
    file_path = "pv_module_efficiency_dataset.csv"
    data = load_data(file_path)
    X_train, X_validation, X_test, y_train_1, y_validation_1, y_test_1, y_train_2, y_validation_2, y_test_2 = split_data(
        data, column_1="expected_efficiency", column_2="efficiency_level"
    )
    categorical_cols = ["pv_module_type","hotspot","birddrop","soiling","junction_box"]
    numerical_cols=["affected_area","temperature","irradiance","Voc","Isc"]
    X_train_processed, X_validation_processed, X_test_processed = preprocess_data(
        X_train, X_validation, X_test, categorical_cols
    )

    # RANDOM FOREST CLASSIFICATION MODEL 
    RF_classif = RandomForestClassifier()
    RF_classif.fit(X_train_processed, y_train_2.values.ravel())
    y_pred_classif = RF_classif.predict(X_test_processed)
    y_pred_classif_proba = RF_classif.predict_proba(X_test_processed)

    #param_grid = {
    #'n_estimators': [50, 100, 200],
    #'max_depth': [10, 20, None],
    #'min_samples_split': [2, 5, 10]
    #}

    #grid_search = GridSearchCV(estimator=RF_classif, param_grid=param_grid, cv=5, scoring='accuracy')  # Usando validação cruzada dentro do grid search
    #grid_search.fit(X_train_processed, y_train_2.values.ravel())
    #print(f"Melhores parâmetros: {grid_search.best_params_}")
    #best_rf = grid_search.best_estimator_
    #y_pred_classif = best_rf.predict(X_test_processed)
    #y_pred_classif_proba = best_rf.predict_proba(X_test_processed)

    #accuracy = accuracy_score(y_test_2, y_pred_classif)
    #print(f"Accuracy no conjunto de validação: {accuracy:.4f}")

    # RANDOM FOREST CLASSIFICATION MODEL - EVALUATION
    #Accuracy
    accuracy = accuracy_score(y_test_2['efficiency_level'], y_pred_classif)
    print(f"Accuracy of Classification Model: {accuracy:.5f}")
    #F1-Score
    f1 = f1_score(y_test_2['efficiency_level'], y_pred_classif, average='weighted')
    print(f"F1 Score of Classification Model: {f1:.5f}")
    #AUC-ROC
    auc_roc = roc_auc_score(y_test_2['efficiency_level'], y_pred_classif_proba, multi_class='ovr', average='weighted')
    print(f"AUC-ROC of Classification Model: {auc_roc:.5f}")
    #Confusion Matrix
    classes = ['extremely_bad', 'bad', 'moderate', 'good']
    conf_matrix = confusion_matrix(y_test_2['efficiency_level'], y_pred_classif, labels=classes)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # RANDOM FOREST REGRESSION MODEL 
    RF_regressor = RandomForestRegressor()
    RF_regressor.fit(X_train_processed, y_train_1.values.ravel())  
    y_pred_reg = RF_regressor.predict(X_test_processed)

    #visualizar os valores previstos e reais para comparar *APAGAR DEPOIS*
    comparison_df = pd.DataFrame({
    'Valor Real': y_test_1.values.ravel(),
    'Valor Previsto': y_pred_reg,
    'Erro': y_test_1.values.ravel() - y_pred_reg
    })
    print(comparison_df.head(30))


    # RANDOM FOREST REGRESSION MODEL - EVALUATION
    errors = y_test_1.values.ravel() - y_pred_reg
    plt.hist(errors, bins=30)
    plt.xlabel('Erro de Previsão')
    plt.ylabel('Frequência')
    plt.title('Distribuição dos Erros de Previsão')
    plt.show()

    mae = mean_absolute_error(y_test_1, y_pred_reg)
    mse = mean_squared_error(y_test_1, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_1, y_pred_reg)
    print(f"Mean Absolute Error (MAE): {mae:.5f}")
    print(f"Mean Squared Error (MSE): {mse:.5f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.5f}")
    print(f"R² (R-squared): {r2:.5f}")
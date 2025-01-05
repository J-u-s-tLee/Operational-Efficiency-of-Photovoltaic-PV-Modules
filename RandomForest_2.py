import pandas as pd
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, mean_squared_error, r2_score, balanced_accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

    X_train_2 = X_train_processed[['affected_area', 'Isc']]
    X_validation_2 = X_validation_processed[['affected_area', 'Isc']]
    X_test_2 = X_test_processed[['affected_area', 'Isc']]

    X_train_val = pd.concat([X_train_2, X_validation_2])
    y_train_val_1 = pd.concat([y_train_1, y_validation_1])
    y_train_val_2 = pd.concat([y_train_2, y_validation_2])

    y_train_val_1 = y_train_val_1.values.ravel()
    y_train_val_2 = y_train_val_2.values.ravel()

    split_index = [-1] * len(X_train_2) + [0] * len(X_validation_2)
    predefined_split = PredefinedSplit(test_fold=split_index)

    # RANDOM FOREST CLASSIFICATION MODEL
    RF_classif = RandomForestClassifier(class_weight='balanced')

    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
    }
    
    grid_search_class = model_selection.GridSearchCV(estimator=RF_classif, param_grid=param_grid, cv=predefined_split, scoring='accuracy')
    grid_search_class.fit(X_train_val, y_train_val_2)
    results = pd.DataFrame(grid_search_class.cv_results_)
    relevant_columns = ['param_n_estimators', 'param_max_depth', 'param_min_samples_split','mean_test_score', 'std_test_score', 'rank_test_score']
    print(results[relevant_columns])
    print("Best Parameters:", grid_search_class.best_params_)
    print("Best Score:", grid_search_class.best_score_)

    best_class = grid_search_class.best_estimator_
    y_pred_classif = best_class.predict(X_test_2)
    y_pred_classif_proba = best_class.predict_proba(X_test_2)

    # RANDOM FOREST CLASSIFICATION MODEL - EVALUATION
    print("RANDOM FOREST CLASSIFICATION MODEL - EVALUATION")
    #Accuracy
    accuracy = accuracy_score(y_test_2['efficiency_level'], y_pred_classif)
    print(f"Accuracy of Classification Model: {accuracy:.5f}")
    #Balanced-Accuracy
    balanced_accuracy = balanced_accuracy_score(y_test_2['efficiency_level'], y_pred_classif)
    print(f"Balanced-accuracy of Classification Model: {balanced_accuracy:.5f}")
    #F1-Score
    f1 = f1_score(y_test_2['efficiency_level'], y_pred_classif, average='weighted')
    print(f"F1 Score of Classification Model: {f1:.5f}")
    #Recall
    recall = recall_score(y_test_2['efficiency_level'], y_pred_classif, average='weighted')
    print(f"Recall: {recall:.5f}")  
    #Precision
    precision = precision_score(y_test_2['efficiency_level'], y_pred_classif, average='weighted', zero_division=0)
    print(f"Precision:{precision:.5f}")
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

    grid_search_reg = model_selection.GridSearchCV(estimator=RF_regressor, param_grid=param_grid, cv=predefined_split, scoring='neg_mean_squared_error')
    grid_search_reg.fit(X_train_val, y_train_val_1)
    results = pd.DataFrame(grid_search_reg.cv_results_)
    relevant_columns = ['param_n_estimators', 'param_max_depth', 'param_min_samples_split','mean_test_score', 'std_test_score', 'rank_test_score']
    print(results[relevant_columns])
    print("Best Parameters:", grid_search_reg.best_params_)
    print("Best Score:", grid_search_reg.best_score_)

    best_class = grid_search_reg.best_estimator_
    y_pred_reg = best_class.predict(X_test_2)

    # RANDOM FOREST REGRESSION MODEL - EVALUATION
    print("RANDOM FOREST REGRESSION MODEL - EVALUATION")
    #Loss
    mse = mean_squared_error(y_test_1, y_pred_reg)
    print(f"Mean Squared Error (MSE): {mse:.5f}")
    #R^2
    r2 = r2_score(y_test_1, y_pred_reg)
    print(f"R² (R-squared): {r2:.5f}")  
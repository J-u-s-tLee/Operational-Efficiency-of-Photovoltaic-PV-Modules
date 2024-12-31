import pandas as pd
import matplotlib.pyplot as plt

def aggregate_importance_by_original_features(importance_df, categorical_cols, numerical_cols, X_data, encoder, save_dir):

    expanded_categorical_cols = encoder.get_feature_names_out(categorical_cols)
    
    feature_mapping = {}
    for cat_col in categorical_cols:
        matching_expanded_cols = [col for col in expanded_categorical_cols if col.startswith(cat_col + "_")]
        for expanded_col in matching_expanded_cols:
            feature_mapping[expanded_col] = cat_col
    
    for num_col in numerical_cols:
        feature_mapping[num_col] = num_col
    
    consolidated_importance = {}
    for expanded_col, importance in zip(importance_df['Feature'], importance_df['Importance']):
        original_feature = feature_mapping.get(expanded_col, expanded_col)

        if original_feature not in consolidated_importance:
            consolidated_importance[original_feature] = 0
        consolidated_importance[original_feature] += importance

    consolidated_df = pd.DataFrame({
        'Feature': consolidated_importance.keys(),
        'Importance': consolidated_importance.values()
    }).sort_values(by='Importance', ascending=False)
    
    top_features = consolidated_df.head(2)['Feature'].tolist()
    print("\nTop 2 most important features:", top_features)

    top_features_df = consolidated_df.head(2) 
    for feature, importance in zip(top_features_df['Feature'], top_features_df['Importance']):
        print(f"Feature: {feature}, Importance: {importance*100:.2f}%")
    

    plt.figure(figsize=(8, 6))
    
    plot_features = []
    for feature in top_features:
        if feature in categorical_cols:
            expanded_cols = [col for col in expanded_categorical_cols if col.startswith(feature)]
            concatenated_data = X_data[expanded_cols].max(axis=1) 
            plot_features.append(concatenated_data)
        else:
            plot_features.append(X_data[feature])

    plt.scatter(plot_features[0], plot_features[1], alpha=0.6, edgecolor='k')
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title('PCA analysis')

    plt.savefig(f"{save_dir}/pca.png")

    return consolidated_df

import pandas as pd
from sklearn.decomposition import PCA


def apply_pca(X, n_components=None):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': pca.components_[0]**2 
    }).sort_values(by='Importance', ascending=False)
    
    return pca, X_pca, importance_df

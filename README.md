# Operational Efficiency of Photovoltaic (PV) Modules

---

## Overview
This repository contains the code and methodology used to evaluate the impact of environmental and manufacturing factors on the operational efficiency of Photovoltaic (PV) modules. The project focuses on utilizing Machine Learning (ML) and Deep Learning (DL) models to forecast both the continuous **expected efficiency** (regression) and the categorical **efficiency level** (classification) of the solar modules.

---

## Dataset Description
The dataset comprises 100,000 observations designed to assist in computer-aided diagnostics (DACO) for PV module performance.

**Features:**
*   **Categorical:** `pv_module_type` (Monocrystalline, Polycrystalline, Thin-Film), `hotspot` (Severe Hotspot, Mild Hotspot, None), `birddrop` (Many, Few, None), `soiling` (High, Moderate, Low, None), and `junction_box` (Visible, None).
*   **Numerical:** `affected_area` (percentage of module's area affected by defects), `temperature` (average temperature), `irradiance` (irradiance level), `Voc` (voltage in open circuit), and `Isc` (current in short circuit).

**Labels:**
*   **expected efficiency**: Continuous value indicating the percentage estimate of the PV module's efficiency (Target for Regression).
*   **efficiency level**: Multiclass categorical variable ('good', 'moderate', 'bad', 'extremely bad') (Target for Classification).

---

## Data Processing
To prepare the dataset for the models, a comprehensive processing pipeline was implemented:
1.  **Categorical Features:** Processed using One-Hot Encoding.
2.  **Numerical Features:** Processed using `MinMaxScaler` (scaling to 0-1) followed by `StandardScaler` (standardizing to zero mean and unit variance).
3.  **Dimensionality Reduction:** Principal Component Analysis (PCA) was used to identify the most significant features. The analysis revealed that `affected_area` and `Isc` (Short-Circuit Current) are the two most important features in the dataset, contributing heavily to the data variance. 

---

## Models Implemented
The models were trained and tested using two different approaches: first with all 10 features, and then with only the 2 most important features identified by the PCA.

1.  **Random Forest (RF):** Utilized for both regression (MSE optimization) and classification (Gini Impurity optimization).
2.  **Support Vector Machine (SVM):** Implemented for multiclass classification using a One-vs-Rest strategy, testing 'linear' and 'rbf' kernels.
3.  **Linear Regression:** Employed exclusively for the regression task using the Ordinary Least Squares method.
4.  **Feed Forward Neural Network (FFNN):** A multi-task deep learning framework utilizing a shared ReLU-activated hidden layer with parallel task-specific output layers. It uses MSE for regression and cross-entropy for classification, incorporating dropout (rate = 0.3) to mitigate overfitting. 

---

## Key Results
*   **Data Analysis:** The dataset presented a severe class imbalance in the categorical label ('good' having the most observations). 
*   **All Features (10):** The FFNN was the top performer, achieving an accuracy of 99.63% in classification and an MSE of 0.0350 with a perfect R-squared for regression. Random forest also performed exceptionally well, achieving a null MSE for regression.
*   **Reduced Features (2):** Feature reduction heavily decreased the models' overall performance. FFNN maintained the highest classification accuracy (76.51%), though SVM demonstrated a better balanced accuracy (69.81%). In regression, the FFNN remained robust while the ML models heavily declined.
*   **Conclusion:** The models evidenced overfitting to the data, likely due to the homogeneous variance across the training, validation, and test sets. 

---

## Academic Context
*Faculdade de Engenharia da Universidade do Porto (FEUP) - Computer-Aided Diagnostics (DACO)*

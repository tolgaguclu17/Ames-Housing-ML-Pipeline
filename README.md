# Ames Housing Price Prediction: End-to-End ML Pipeline

## Project Overview
This project builds a robust, production-ready machine learning pipeline to predict residential property prices in Ames, Iowa. Moving beyond basic modeling, this repository demonstrates advanced feature engineering, strict data-leakage prevention, and target transformation techniques to handle highly skewed real-world financial data.

## Key Architectural Highlights

* **Leak-Free Custom Transformers:** Implemented a custom `FeatureEngineer` class inheriting from Scikit-Learn's `BaseEstimator` and `TransformerMixin`. This allows the pipeline to accept raw, uncleaned data directly in production while ensuring all spatial, temporal, and logical feature creations happen securely within cross-validation folds.
* **Asymmetric Target Management:** Utilized `TransformedTargetRegressor` with `np.log1p` to automatically normalize the right-skewed `SalePrice` during training and return predictions in actual dollar amounts, stabilizing the model's error gradients.
* **Robust Outlier Strategy:** Extreme outliers (e.g., houses >4000 sqft with anomalous sale conditions) were filtered *exclusively* from the training set to prevent model distortion, preserving the test set's integrity as a representation of real-world scenarios.
* **Differentiated Preprocessing:** Built a branching `ColumnTransformer` that applies `StandardScaler` to linear models while bypassing it for tree-based models, optimizing the input for different algorithmic families simultaneously.

## Model Performance & Selection

A comprehensive baseline comparison evaluated Dummy, Ridge, Lasso, ElasticNet, Gradient Boosting, XGBoost, and LightGBM models using Repeated 5-Fold Cross-Validation.

The linear models, bolstered by robust scaling and target transformation, outperformed the tree-based ensembles. **ElasticNet** was selected as the final model after hyperparameter tuning via `RandomizedSearchCV`.

**Final Test Set Metrics:**
* **R² Score:** `0.9393`
* **RMSLE:** `0.1285` *(Primary optimization metric for multiplicative price variance)*
* **MAE:** `$14,238.40`
* **RMSE:** `$21,572.60`

## Repository Structure

\`\`\`text
ames-housing-prediction/
├── data/
│   ├── raw/                        # Original Kaggle datasets
│   └── processed/                  # Transformed data artifacts
├── models/
│   └── ames_housing_model.pkl      # Serialized, production-ready pipeline
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis & statistical testing
│   └── 02_Modeling_Pipeline.ipynb  # Preprocessing, tuning, and evaluation
├── .gitignore
├── README.md
└── requirements.txt                # Project dependencies
\`\`\`

## How to Run

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/YOUR_USERNAME/ames-housing-prediction.git
   cd ames-housing-prediction
   \`\`\`
2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
3. Run the Jupyter Notebooks sequentially in the `/notebooks` directory.D
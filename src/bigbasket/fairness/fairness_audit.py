# # src/bigbasket/fairness/fairness_audit.py
# import os
# import joblib
# import pandas as pd
# import matplotlib.pyplot as plt

# MODEL_DIR = os.getenv("MODEL_DIR", "models")
# PROCESSED_DIR = os.getenv("OUT_DIR", "data/processed")
# OUT_DIR = os.getenv("FAIRNESS_DIR", "reports/fairness")
# os.makedirs(OUT_DIR, exist_ok=True)

# def detect_group_column(df):
#     """
#     Automatically detect a categorical column for fairness audit.
#     Preference: 'hotel', 'market_segment', 'distribution_channel'
#     """
#     preferred_cols = ['hotel', 'market_segment', 'distribution_channel']
#     for col in preferred_cols:
#         if col in df.columns:
#             return col
#     # fallback: pick first categorical column (object dtype)
#     categorical_cols = df.select_dtypes(include='object').columns.tolist()
#     if categorical_cols:
#         print(f"No preferred column found, using '{categorical_cols[0]}' for grouping")
#         return categorical_cols[0]
#     raise KeyError(f"No suitable categorical column found in X_test. Available columns: {df.columns.tolist()}")

# def compute_fairness_metrics(model, X_test, y_test, group_col):
#     """Compute group-wise fairness metrics."""
#     df = X_test.copy()
#     df['y_true'] = y_test
#     df['y_pred'] = model.predict(X_test)
    
#     # Accuracy per group
#     acc = df.groupby(group_col).apply(lambda x: (x['y_pred'] == x['y_true']).mean())
    
#     # Positive prediction rate per group
#     pos_rate = df.groupby(group_col)['y_pred'].mean()
    
#     # Error rate per group
#     err_rate = 1 - acc
    
#     metrics_df = pd.DataFrame({
#         "accuracy": acc,
#         "positive_rate": pos_rate,
#         "error_rate": err_rate
#     })
    
#     return metrics_df

# def plot_fairness(metrics_df, group_col):
#     """Plot bar charts for accuracy, positive_rate, and error_rate."""
#     for metric in metrics_df.columns:
#         plt.figure(figsize=(8,5))
#         metrics_df[metric].plot(kind='bar', color='skyblue')
#         plt.title(f"{metric.replace('_', ' ').title()} by {group_col}")
#         plt.ylabel(metric)
#         plt.xlabel(group_col)
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUT_DIR, f"{metric}_by_{group_col}.png"))
#         plt.close()
#     print(f"Saved fairness plots to {OUT_DIR}")

# if __name__ == "__main__":
#     # Load trained model
#     model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
    
#     # Load test data
#     X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
#     y_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet"))
    
#     # Ensure y_test is a Series
#     if isinstance(y_test, pd.DataFrame):
#         y_test = y_test.iloc[:,0]
    
#     # Detect a suitable categorical column automatically
#     group_col = detect_group_column(X_test)
#     print(f"Using '{group_col}' as grouping column for fairness audit.")
    
#     # Compute metrics
#     metrics_df = compute_fairness_metrics(model, X_test, y_test, group_col=group_col)
    
#     # Save metrics to CSV
#     metrics_df.to_csv(os.path.join(OUT_DIR, "group_fairness_metrics.csv"))
#     print(f"Saved fairness metrics CSV to {OUT_DIR}")
    
#     # Generate plots
#     plot_fairness(metrics_df, group_col=group_col)

# src/hotel/fairness/fairness_audit.py
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Directories (relative paths)
MODEL_DIR = "models"
PROCESSED_DIR = "data/processed"
OUT_DIR = "reports/fairness"
os.makedirs(OUT_DIR, exist_ok=True)

def detect_group_column(df):
    preferred_cols = ['hotel', 'market_segment', 'distribution_channel']
    for col in preferred_cols:
        if col in df.columns:
            return col
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        print(f"No preferred column found, using '{categorical_cols[0]}' for grouping")
        return categorical_cols[0]
    raise KeyError(f"No suitable categorical column found in X_test. Available columns: {df.columns.tolist()}")

def compute_fairness_metrics(model, X_test, y_test, group_col):
    df = X_test.copy()
    df['y_true'] = y_test
    df['y_pred'] = model.predict(X_test)

    acc = df.groupby(group_col, group_keys=False).apply(lambda x: (x['y_pred'] == x['y_true']).mean())
    pos_rate = df.groupby(group_col, group_keys=False)['y_pred'].mean()
    err_rate = 1 - acc

    metrics_df = pd.DataFrame({
        "accuracy": acc,
        "positive_rate": pos_rate,
        "error_rate": err_rate
    })
    return metrics_df

def plot_fairness(metrics_df, group_col):
    for metric in metrics_df.columns:
        plt.figure(figsize=(8,5))
        metrics_df[metric].plot(kind='bar', color='skyblue')
        plt.title(f"{metric.replace('_', ' ').title()} by {group_col}")
        plt.ylabel(metric)
        plt.xlabel(group_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{metric}_by_{group_col}.png"))
        plt.close()
    print(f"Saved fairness plots to {OUT_DIR}")

if __name__ == "__main__":
    model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
    X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet"))

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:,0]

    group_col = detect_group_column(X_test)
    print(f"Using '{group_col}' as grouping column for fairness audit.")

    metrics_df = compute_fairness_metrics(model, X_test, y_test, group_col=group_col)
    metrics_df.to_csv(os.path.join(OUT_DIR, "group_fairness_metrics.csv"), index=True)
    print(f"Saved fairness metrics CSV to {OUT_DIR}")

    plot_fairness(metrics_df, group_col)

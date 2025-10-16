# # src/bigbasket/explain/explainability.py
# import joblib
# import pandas as pd
# import shap
# import os
# from lime.lime_tabular import LimeTabularExplainer
# import matplotlib.pyplot as plt
# from sklearn.feature_selection import VarianceThreshold

# MODEL_DIR = os.getenv("MODEL_DIR", "models")
# PROCESSED_DIR = os.getenv("OUT_DIR", "data/processed")
# OUT_DIR = os.getenv("EXPLAIN_DIR", "reports/explain")
# os.makedirs(OUT_DIR, exist_ok=True)

# def global_shap(model, X_sample):
#     # if tree-based use TreeExplainer
#     try:
#         explainer = shap.TreeExplainer(model)
#     except Exception:
#         explainer = shap.KernelExplainer(model.predict_proba, X_sample)
#     shap_values = explainer.shap_values(X_sample)
#     # summary plot
#     plt.figure()
#     shap.summary_plot(shap_values, X_sample, show=False)
#     plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), bbox_inches='tight')
#     print("Saved SHAP summary to", OUT_DIR)

# def clean_for_lime(X_train, X_test, near_zero_threshold=1e-5):
#     """
#     Clean data for LIME:
#     - Fill NaNs with 0
#     - Remove zero or near-zero variance columns
#     """
#     # Fill NaNs
#     X_train_clean = X_train.fillna(0)
#     X_test_clean = X_test.fillna(0)

#     # Remove zero or near-zero variance columns
#     selector = VarianceThreshold(threshold=near_zero_threshold)
#     X_train_clean = pd.DataFrame(selector.fit_transform(X_train_clean),
#                                  columns=X_train.columns[selector.get_support()])
#     X_test_clean = pd.DataFrame(selector.transform(X_test_clean),
#                                 columns=X_train.columns[selector.get_support()])

#     # Warn if any columns were dropped
#     dropped_cols = set(X_train.columns) - set(X_train_clean.columns)
#     if dropped_cols:
#         print(f"Removed {len(dropped_cols)} near-constant columns: {dropped_cols}")

#     return X_train_clean, X_test_clean

# def lime_local(model, X_train, X_sample):
#     # Fill NaNs only, do NOT remove any columns
#     X_train_clean = X_train.fillna(0)
#     X_sample_clean = X_sample.fillna(0)
    
#     explainer = LimeTabularExplainer(
#         training_data=X_train_clean.values,
#         feature_names=X_train_clean.columns.tolist(),
#         class_names=[str(c) for c in model.classes_],
#         mode='classification'
#     )
    
#     exp = explainer.explain_instance(
#         X_sample_clean.values[0], 
#         model.predict_proba, 
#         num_features=10
#     )
    
#     html = exp.as_html()
#     with open(os.path.join(OUT_DIR, "lime_local.html"), "w", encoding="utf-8") as f:
#         f.write(html)
#     print("Saved LIME html to", OUT_DIR)



# if __name__ == "__main__":
#     model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
#     X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
#     X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    
#     # Use a small sample for SHAP to be faster
#     X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    
#     global_shap(model, X_sample)
    
#     # Use first row of X_test for LIME
#     lime_local(model, X_train, X_test.head(1))
# src/hotel/explain/explainability.py
# src/bigbasket/explain/explainability.py
import joblib
import pandas as pd
import shap
import os
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

MODEL_DIR = os.getenv("MODEL_DIR", "models")
PROCESSED_DIR = os.getenv("OUT_DIR", "data/processed")
OUT_DIR = os.getenv("EXPLAIN_DIR", "reports/explain")
os.makedirs(OUT_DIR, exist_ok=True)

def global_shap(model, X_sample):
    """Compute SHAP values and save summary plot."""
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, X_sample)

    shap_values = explainer.shap_values(X_sample)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), bbox_inches='tight')
    print("Saved SHAP summary to", OUT_DIR)

def lime_local(model, X_train, X_sample):
    """Compute LIME local explanations and save HTML."""
    # Remove constant columns to avoid LIME domain errors
    X_train_clean = X_train.loc[:, X_train.nunique() > 1]
    removed_cols = X_train.columns[X_train.nunique() == 1].tolist()
    if removed_cols:
        print("Removed constant columns for LIME:", removed_cols)
    X_sample_clean = X_sample[X_train_clean.columns]

    explainer = LimeTabularExplainer(
        training_data=X_train_clean.values,
        feature_names=X_train_clean.columns.tolist(),
        class_names=[str(c) for c in model.classes_],
        mode='classification'
    )

    try:
        exp = explainer.explain_instance(
            X_sample_clean.values[0],
            model.predict_proba,
            num_features=10
        )
        html = exp.as_html()
        with open(os.path.join(OUT_DIR, "lime_local.html"), "w", encoding="utf-8") as f:
            f.write(html)
        print("Saved LIME html to", OUT_DIR)
    except ValueError as e:
        print("LIME failed:", e)

if __name__ == "__main__":
    model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))

    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))

    # Sample for SHAP to speed up computation
    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)

    global_shap(model, X_sample)
    lime_local(model, X_train, X_test.head(1))
